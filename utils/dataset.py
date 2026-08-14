import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from utils.augmentations import DetectionTransforms


CLASSES = ["person", "car", "dog", "cat", "chair"]
CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(CLASSES)}
IDX_TO_CLASS = {idx: cls_name for idx, cls_name in enumerate(CLASSES)}


class ObjectDetectionDataset(Dataset):
    """
    Dataset loader for Object Detection using JSON annotations.
    """

    def __init__(self, annotation_file: str, image_dir: str, transforms=None):
        self.image_dir = image_dir
        self.transforms = transforms

        with open(annotation_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.classes = data.get("classes", CLASSES)
        self.images_info = {img["id"]: img for img in data["images"]}

        # Group annotations by image_id
        self.annotations = {}
        for ann in data.get("annotations", []):
            img_id = ann["image_id"]
            if img_id not in self.annotations:
                self.annotations[img_id] = []
            self.annotations[img_id].append(ann)

        self.image_ids = list(self.images_info.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.images_info[img_id]

        # Determine full image path flexibly
        file_name = img_info.get("file_name", img_id)
        image_path = os.path.join(self.image_dir, os.path.basename(file_name))
        if not os.path.exists(image_path):
            image_path = os.path.join(self.image_dir, file_name)

        image = Image.open(image_path).convert("RGB")

        # Parse ground-truth boxes and labels
        anns = self.annotations.get(img_id, [])
        boxes = []
        labels = []

        for ann in anns:
            cls_name = ann["class"]
            if cls_name in CLASS_TO_IDX:
                label = CLASS_TO_IDX[cls_name]
                bbox = ann["bbox"]  # [xmin, ymin, xmax, ymax]
                boxes.append(bbox)
                labels.append(label)

        if len(boxes) > 0:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)

        if self.transforms is not None:
            image_tensor, boxes_tensor, labels_tensor, orig_shape = self.transforms(
                image, boxes_tensor, labels_tensor
            )
        else:
            orig_shape = (image.height, image.width)
            image_tensor = torch.tensor(list(image.getdata()), dtype=torch.float32)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": img_id,
            "orig_shape": orig_shape,
        }

        return image_tensor, target


def detection_collate_fn(batch):
    """
    Custom collate function for object detection batching.
    Handles variable-size images from multi-scale training by padding to max size.
    batch: list of tuples (image_tensor, target_dict)
    """
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    # Check if all images are the same size (common case)
    sizes = set((img.shape[1], img.shape[2]) for img in images)
    if len(sizes) == 1:
        images = torch.stack(images, dim=0)
    else:
        # Pad to the max H and W in the batch
        max_h = max(img.shape[1] for img in images)
        max_w = max(img.shape[2] for img in images)
        padded = []
        for img in images:
            pad_h = max_h - img.shape[1]
            pad_w = max_w - img.shape[2]
            padded_img = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h), value=0)
            padded.append(padded_img)
        images = torch.stack(padded, dim=0)

    return images, targets

