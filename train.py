import os
import json
import argparse
import subprocess
import time
from datetime import timedelta
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from models.retinanet import RetinaNet
from utils.dataset import ObjectDetectionDataset, IDX_TO_CLASS, detection_collate_fn
from utils.augmentations import DetectionTransforms


def parse_args():
    parser = argparse.ArgumentParser(description="RetinaNet Training Script from Scratch (Supports Multi-GPU DDP)")
    parser.add_argument("--train_data", type=str, default="./public/annotations/train.json", help="Path to train annotation JSON")
    parser.add_argument("--val_data", type=str, default="./public/annotations/val.json", help="Path to val annotation JSON")
    parser.add_argument("--image_dir", type=str, default="./public/train/images", help="Path to train images directory")
    parser.add_argument("--val_image_dir", type=str, default="./public/val/images", help="Path to val images directory")
    parser.add_argument("--checkpoint_dir", type=str, default="./models/", help="Directory to save checkpoints")

    # Training Hyperparameters
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=1e-4, help="Base learning rate for heads/FPN")
    parser.add_argument("--backbone_lr_ratio", type=float, default=0.1, help="LR multiplier for backbone fine-tuning")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--backbone", type=str, default="resnet50", choices=["resnet34", "resnet50"], help="Backbone architecture")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader num workers")
    parser.add_argument("--img_size", type=int, default=640, help="Target image square size for training")

    # Advanced training options
    parser.add_argument("--warmup_iters", type=int, default=500, help="Number of warmup iterations for LR")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Max gradient norm for clipping")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# DDP Helpers
# ---------------------------------------------------------------------------

def is_main_process(rank):
    return rank == 0


def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_distributed = world_size > 1

    if is_distributed:
        torch.cuda.set_device(local_rank)
        # Set 30-minute timeout for NCCL collective operations to prevent timeout during validation
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return is_distributed, rank, local_rank, world_size, device


def cleanup_distributed(is_distributed):
    if is_distributed and dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Learning Rate Warmup Scheduler (Multi-Parameter Groups)
# ---------------------------------------------------------------------------

class WarmupCosineScheduler:
    """
    Linear warmup for `warmup_iters` steps, then cosine annealing.
    Supports differential learning rates across multiple parameter groups.
    """

    def __init__(self, optimizer, warmup_iters, total_iters, base_lrs=None):
        self.optimizer = optimizer
        self.warmup_iters = warmup_iters
        self.total_iters = total_iters
        if base_lrs is None:
            self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        else:
            self.base_lrs = base_lrs
        self.current_iter = 0

    def step(self):
        self.current_iter += 1
        if self.current_iter <= self.warmup_iters:
            factor = self.current_iter / max(self.warmup_iters, 1)
        else:
            progress = (self.current_iter - self.warmup_iters) / max(self.total_iters - self.warmup_iters, 1)
            import math
            factor = 0.5 * (1.0 + math.cos(math.pi * progress))

        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group["lr"] = base_lr * factor


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train_one_epoch(model, dataloader, optimizer, scaler, device, epoch, scheduler_iter, grad_clip, rank=0):
    model.train()
    running_loss = 0.0
    running_cls_loss = 0.0
    running_reg_loss = 0.0
    start_time = time.time()

    for step, (images, targets) in enumerate(dataloader):
        images = images.to(device)
        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            loss_dict = model(images, targets)
            loss = loss_dict["loss"]
            cls_loss = loss_dict["loss_cls"]
            reg_loss = loss_dict["loss_reg"]

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

        if scheduler_iter is not None:
            scheduler_iter.step()

        running_loss += loss.item()
        running_cls_loss += cls_loss.item()
        running_reg_loss += reg_loss.item()

        if is_main_process(rank) and ((step + 1) % 20 == 0 or (step + 1) == len(dataloader)):
            elapsed = time.time() - start_time
            head_lr = optimizer.param_groups[1]["lr"] if len(optimizer.param_groups) > 1 else optimizer.param_groups[0]["lr"]
            print(
                f"  Epoch [{epoch+1}] Step [{step+1}/{len(dataloader)}] "
                f"Loss: {loss.item():.4f} (Cls: {cls_loss.item():.4f}, Reg: {reg_loss.item():.4f}) "
                f"LR: {head_lr:.6f} Time: {elapsed:.1f}s"
            )

    epoch_loss = running_loss / len(dataloader)
    epoch_cls = running_cls_loss / len(dataloader)
    epoch_reg = running_reg_loss / len(dataloader)
    return epoch_loss, epoch_cls, epoch_reg


# ---------------------------------------------------------------------------
# Validation — compute val loss
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate_loss(model, dataloader, device):
    model.eval()
    val_loss = 0.0
    val_cls = 0.0
    val_reg = 0.0

    for images, targets in dataloader:
        images = images.to(device)
        loss_dict = model(images, targets)
        val_loss += loss_dict["loss"].item()
        val_cls += loss_dict["loss_cls"].item()
        val_reg += loss_dict["loss_reg"].item()

    n = max(len(dataloader), 1)
    return val_loss / n, val_cls / n, val_reg / n


# ---------------------------------------------------------------------------
# Validation — compute mAP
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_map(model, dataloader, device, val_ann_path, img_size, output_dir):
    raw_model = model.module if hasattr(model, "module") else model
    raw_model.eval()

    predictions = []
    for images, targets in dataloader:
        images = images.to(device)
        batch_results = raw_model(images)

        for i, det in enumerate(batch_results):
            img_id = targets[i]["image_id"]
            orig_h, orig_w = targets[i]["orig_shape"]

            boxes = det["boxes"].cpu()
            scores = det["scores"].cpu()
            labels = det["labels"].cpu()

            boxes_list = []
            if len(boxes) > 0:
                scale_x = orig_w / float(img_size)
                scale_y = orig_h / float(img_size)
                boxes[:, 0] = (boxes[:, 0] * scale_x).clamp(min=0, max=orig_w)
                boxes[:, 1] = (boxes[:, 1] * scale_y).clamp(min=0, max=orig_h)
                boxes[:, 2] = (boxes[:, 2] * scale_x).clamp(min=0, max=orig_w)
                boxes[:, 3] = (boxes[:, 3] * scale_y).clamp(min=0, max=orig_h)

                for box, score, label_idx in zip(boxes, scores, labels):
                    cls_name = IDX_TO_CLASS[label_idx.item()]
                    xmin, ymin, xmax, ymax = box.tolist()
                    boxes_list.append({
                        "class": cls_name,
                        "confidence": round(float(score.item()), 4),
                        "bbox": [round(xmin, 2), round(ymin, 2), round(xmax, 2), round(ymax, 2)],
                    })

            predictions.append({"image_id": img_id, "boxes": boxes_list})

    pred_path = os.path.join(output_dir, "_val_predictions_tmp.json")
    score_path = os.path.join(output_dir, "_val_score_tmp.json")
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    eval_script = os.path.join("public", "tools", "evaluate_predictions.py")
    if not os.path.exists(eval_script):
        eval_script = os.path.join(".", "public", "tools", "evaluate_predictions.py")

    if os.path.exists(eval_script):
        try:
            result = subprocess.run(
                ["python", eval_script,
                 "--ground_truth", val_ann_path,
                 "--predictions", pred_path,
                 "--output", score_path],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode == 0 and os.path.exists(score_path):
                with open(score_path, "r") as f:
                    score_data = json.load(f)
                mAP = score_data.get("mAP@0.5", score_data.get("mAP", 0.0))
                return float(mAP)
        except Exception as e:
            print(f"  [WARN] Could not run evaluate_predictions.py: {e}")

    return None


# ---------------------------------------------------------------------------
# Main Entry
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    is_distributed, rank, local_rank, world_size, device = setup_distributed()

    if is_main_process(rank):
        os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_ckpt_path = os.path.join(args.checkpoint_dir, "best.pth")

    if is_main_process(rank):
        print(f"Device: {device} | Distributed: {is_distributed} (World Size: {world_size})")

    # Build Data Pipelines
    train_transforms = DetectionTransforms(
        target_size=(args.img_size, args.img_size), is_train=True, multi_scale=True
    )
    val_transforms = DetectionTransforms(
        target_size=(args.img_size, args.img_size), is_train=False, multi_scale=False
    )

    train_dataset = ObjectDetectionDataset(args.train_data, args.image_dir, transforms=train_transforms)
    val_dataset = ObjectDetectionDataset(args.val_data, args.val_image_dir, transforms=val_transforms)

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if is_distributed else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=detection_collate_fn,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=detection_collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    if is_main_process(rank):
        print(f"Loaded {len(train_dataset)} training samples, {len(val_dataset)} validation samples.")

    # Initialize RetinaNet Model
    model = RetinaNet(num_classes=5, backbone_name=args.backbone, pretrained=True).to(device)

    # Wrap model with DDP if multi-GPU
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    # Differential Learning Rates: Backbone gets 10x smaller LR for safe fine-tuning
    raw_model = model.module if is_distributed else model
    backbone_params = list(raw_model.backbone.parameters())
    head_params = [p for n, p in raw_model.named_parameters() if not n.startswith("backbone")]

    effective_base_lr = args.lr * world_size
    effective_backbone_lr = effective_base_lr * args.backbone_lr_ratio

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": effective_backbone_lr},
            {"params": head_params, "lr": effective_base_lr},
        ],
        weight_decay=args.weight_decay,
    )

    total_iters = args.epochs * len(train_loader)
    scheduler_iter = WarmupCosineScheduler(
        optimizer,
        warmup_iters=args.warmup_iters,
        total_iters=total_iters,
        base_lrs=[effective_backbone_lr, effective_base_lr],
    )

    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    best_metric = -1.0
    best_val_loss = float("inf")
    use_map_tracking = True

    if is_main_process(rank):
        print("=" * 70)
        print("Starting training pipeline...")
        print(f"  GPUs: {world_size}, Batch per GPU: {args.batch_size} (Total Batch: {args.batch_size * world_size})")
        print(f"  Epochs: {args.epochs}")
        print(f"  Heads Base LR: {effective_base_lr:.6f} | Backbone LR: {effective_backbone_lr:.6f}")
        print(f"  Warmup iters: {args.warmup_iters}, Grad clip: {args.grad_clip}")
        print(f"  Multi-scale training: ON, Image size: {args.img_size}")
        print("=" * 70)

    for epoch in range(args.epochs):
        if is_distributed:
            train_sampler.set_epoch(epoch)

        if is_main_process(rank):
            print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")

        train_loss, train_cls, train_reg = train_one_epoch(
            model, train_loader, optimizer, scaler, device, epoch,
            scheduler_iter, args.grad_clip, rank=rank,
        )

        # Validation & checkpointing only on main process
        if is_main_process(rank):
            val_loss, val_cls, val_reg = validate_loss(model, val_loader, device)

            print(
                f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} "
                f"(Val Cls: {val_cls:.4f}, Val Reg: {val_reg:.4f})"
            )

            val_map = evaluate_map(
                model, val_loader, device, args.val_data, args.img_size, args.checkpoint_dir
            )

            should_save = False

            if val_map is not None:
                print(f"  Val mAP@0.5: {val_map:.4f}")
                if val_map > best_metric:
                    best_metric = val_map
                    should_save = True
                    print(f"  ★ New best mAP@0.5: {val_map:.4f}")
            else:
                if use_map_tracking:
                    print("  [INFO] mAP evaluation unavailable, falling back to val loss tracking.")
                    use_map_tracking = False
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    should_save = True
                    print(f"  ★ New best val loss: {val_loss:.4f}")

            if should_save:
                raw_model = model.module if hasattr(model, "module") else model
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": raw_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_val_loss": val_loss,
                        "best_map": val_map if val_map is not None else -1,
                        "args": vars(args),
                    },
                    best_ckpt_path,
                )
                print(f"  → Saved best checkpoint to {best_ckpt_path}")

        # Synchronize ranks across epochs safely
        if is_distributed:
            dist.barrier()

    if is_main_process(rank):
        print("=" * 70)
        print(f"Training completed! Best checkpoint: {best_ckpt_path}")
        if best_metric > 0:
            print(f"Best mAP@0.5: {best_metric:.4f}")
        print("=" * 70)

    cleanup_distributed(is_distributed)


if __name__ == "__main__":
    main()
