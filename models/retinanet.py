import torch
import torch.nn as nn
from models.backbone import ResNetBackbone
from models.fpn import FeaturePyramidNetwork
from models.anchor import AnchorGenerator, AnchorMatcher
from models.head import ClassificationHead, RegressionHead
from models.losses import FocalLoss, BBoxRegressionLoss
from utils.box_utils import box_transform, box_decode, clip_boxes
from utils.nms import batched_nms


class RetinaNet(nn.Module):
    """
    RetinaNet Object Detector written from scratch in PyTorch.
    Combines ResNet Backbone, FPN, Multi-level Anchors, Dual Heads, Focal Loss, and Per-class NMS.
    """

    def __init__(
        self,
        num_classes=5,
        backbone_name="resnet50",
        pretrained=True,
        conf_threshold=0.05,
        nms_threshold=0.5,
        max_detections_per_img=100,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.max_detections_per_img = max_detections_per_img

        # 1. Feature Extractor
        self.backbone = ResNetBackbone(architecture=backbone_name, pretrained=pretrained)
        self.fpn = FeaturePyramidNetwork(in_channels_list=self.backbone.out_channels, out_channels=256)

        # 2. Anchors & Matching
        self.anchor_generator = AnchorGenerator()
        self.anchor_matcher = AnchorMatcher(pos_threshold=0.5, neg_threshold=0.4)

        # 3. Heads
        num_anchors = self.anchor_generator.num_anchors_per_location
        self.cls_head = ClassificationHead(in_channels=256, num_anchors=num_anchors, num_classes=num_classes)
        self.reg_head = RegressionHead(in_channels=256, num_anchors=num_anchors)

        # 4. Losses
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0, reduction="sum")
        self.reg_loss = BBoxRegressionLoss(reduction="sum")

    def forward(self, images: torch.Tensor, targets: list = None):
        """
        images: Tensor (B, 3, H, W)
        targets: List of dicts containing 'boxes' (N, 4) and 'labels' (N,)
        """
        batch_size, _, img_h, img_w = images.shape

        # Extract features
        backbone_feats = self.backbone(images)
        pyramid_feats = self.fpn(backbone_feats)

        # Generate predictions
        cls_logits = self.cls_head(pyramid_feats)  # (B, N_total, K)
        reg_deltas = self.reg_head(pyramid_feats)  # (B, N_total, 4)

        # Generate anchors
        anchors = self.anchor_generator(pyramid_feats, (img_h, img_w))  # (N_total, 4)

        if targets is not None:
            return self._compute_losses(cls_logits, reg_deltas, anchors, targets)
        else:
            return self._predict_boxes(cls_logits, reg_deltas, anchors, (img_h, img_w))

    def _compute_losses(self, cls_logits: torch.Tensor, reg_deltas: torch.Tensor, anchors: torch.Tensor, targets: list):
        batch_size = cls_logits.shape[0]
        device = cls_logits.device
        total_cls_loss = torch.tensor(0.0, device=device)
        total_reg_loss = torch.tensor(0.0, device=device)
        total_positives = 0

        for b in range(batch_size):
            gt_boxes = targets[b]["boxes"].to(images_device := cls_logits.device)
            gt_labels = targets[b]["labels"].to(images_device)

            matched_gt_boxes, matched_labels = self.anchor_matcher.match(anchors, gt_boxes, gt_labels)

            pos_mask = matched_labels >= 0
            valid_mask = matched_labels != -2  # Exclude ignored anchors

            num_pos = pos_mask.sum().item()
            total_positives += num_pos

            # Compute Focal Loss for image b
            b_cls_loss = self.focal_loss(cls_logits[b], matched_labels, valid_mask)
            total_cls_loss += b_cls_loss

            # Compute Regression Loss strictly for positive anchors in image b
            if num_pos > 0:
                pos_anchors = anchors[pos_mask]
                pos_matched_gt = matched_gt_boxes[pos_mask]
                target_deltas = box_transform(pos_anchors, pos_matched_gt)
                b_reg_loss = self.reg_loss(reg_deltas[b][pos_mask], target_deltas)
                total_reg_loss += b_reg_loss

        normalizer = max(float(total_positives), 1.0)
        loss_cls = total_cls_loss / normalizer
        loss_reg = total_reg_loss / normalizer
        total_loss = loss_cls + loss_reg

        return {
            "loss_cls": loss_cls,
            "loss_reg": loss_reg,
            "loss": total_loss,
        }

    @torch.no_grad()
    def _predict_boxes(self, cls_logits: torch.Tensor, reg_deltas: torch.Tensor, anchors: torch.Tensor, img_shape: tuple):
        batch_size = cls_logits.shape[0]
        results = []

        cls_probs = torch.sigmoid(cls_logits)  # (B, N_total, K)

        for b in range(batch_size):
            b_probs = cls_probs[b]  # (N_total, K)
            b_deltas = reg_deltas[b]  # (N_total, 4)

            # Filter predictions above confidence threshold
            keep_mask = b_probs > self.conf_threshold

            if not keep_mask.any():
                results.append({
                    "boxes": torch.empty((0, 4), device=cls_logits.device),
                    "scores": torch.empty((0,), device=cls_logits.device),
                    "labels": torch.empty((0,), dtype=torch.int64, device=cls_logits.device),
                })
                continue

            anchor_indices, class_indices = torch.where(keep_mask)

            scores = b_probs[anchor_indices, class_indices]
            selected_anchors = anchors[anchor_indices]
            selected_deltas = b_deltas[anchor_indices]

            # Decode boxes
            decoded_boxes = box_decode(selected_anchors, selected_deltas)
            decoded_boxes = clip_boxes(decoded_boxes, img_shape)

            # Keep top 1000 candidate detections per image before NMS for fast processing
            if len(scores) > 1000:
                topk_scores, topk_idx = torch.topk(scores, 1000)
                decoded_boxes = decoded_boxes[topk_idx]
                scores = topk_scores
                class_indices = class_indices[topk_idx]

            # Apply Per-class Non-Maximum Suppression (NMS)
            keep_nms_indices = batched_nms(
                decoded_boxes, scores, class_indices, self.nms_threshold
            )

            # Limit max detections per image
            if len(keep_nms_indices) > self.max_detections_per_img:
                keep_nms_indices = keep_nms_indices[: self.max_detections_per_img]

            final_boxes = decoded_boxes[keep_nms_indices]
            final_scores = scores[keep_nms_indices]
            final_labels = class_indices[keep_nms_indices]

            results.append({
                "boxes": final_boxes,
                "scores": final_scores,
                "labels": final_labels,
            })

        return results
