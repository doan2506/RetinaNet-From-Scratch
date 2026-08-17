import torch
import torch.nn as nn
from utils.box_utils import box_iou


class AnchorGenerator(nn.Module):
    """
    Generates anchor boxes across feature pyramid maps P3, P4, P5, P6, P7.
    """

    def __init__(
        self,
        strides=(8, 16, 32, 64, 128),
        base_sizes=(32, 64, 128, 256, 512),
        ratios=(0.5, 1.0, 2.0),
        scales=(1.0, 2 ** (1 / 3), 2 ** (2 / 3)),
    ):
        super().__init__()
        self.strides = strides
        self.base_sizes = base_sizes
        self.register_buffer("ratios", torch.tensor(ratios, dtype=torch.float32))
        self.register_buffer("scales", torch.tensor(scales, dtype=torch.float32))
        self.num_anchors_per_location = len(ratios) * len(scales)

    def _generate_base_anchors(self, base_size: float, device: torch.device) -> torch.Tensor:
        """
        Generates 9 base anchors centered at (0, 0) for a given base size.
        Returns: Tensor (9, 4) in [xmin, ymin, xmax, ymax]
        """
        ratios = self.ratios
        scales = self.scales

        # Calculate heights and widths for all (scale, ratio) combinations
        aspect_ratios = torch.sqrt(ratios)
        h = (base_size * scales[:, None] / aspect_ratios[None, :]).view(-1)
        w = (base_size * scales[:, None] * aspect_ratios[None, :]).view(-1)

        # Centered at (0, 0)
        xmin = -0.5 * w
        ymin = -0.5 * h
        xmax = 0.5 * w
        ymax = 0.5 * h

        return torch.stack([xmin, ymin, xmax, ymax], dim=1)

    def forward(self, feature_maps: list, image_size: tuple) -> torch.Tensor:
        """
        feature_maps: List of feature map Tensors [P3, P4, P5, P6, P7]
        image_size: (height, width) tuple
        Returns: Tensor of all concatenated anchors across all pyramid levels (N_total, 4)
        """
        device = feature_maps[0].device
        all_anchors = []

        for feature_map, stride, base_size in zip(feature_maps, self.strides, self.base_sizes):
            _, _, feat_h, feat_w = feature_map.shape
            base_anchors = self._generate_base_anchors(base_size, device)  # (9, 4)

            # Grid coordinates centered in image pixels
            shift_x = (torch.arange(0, feat_w, device=device, dtype=torch.float32) + 0.5) * stride
            shift_y = (torch.arange(0, feat_h, device=device, dtype=torch.float32) + 0.5) * stride

            shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing="ij")
            shifts = torch.stack([shift_x.reshape(-1), shift_y.reshape(-1),
                                  shift_x.reshape(-1), shift_y.reshape(-1)], dim=1)  # (H*W, 4)

            # Add shifts to base anchors -> (H*W, 9, 4) -> (H*W*9, 4)
            anchors = (base_anchors[None, :, :] + shifts[:, None, :]).reshape(-1, 4)
            all_anchors.append(anchors)

        return torch.cat(all_anchors, dim=0)


class AnchorMatcher:
    """
    Matches generated anchors with Ground Truth bounding boxes.
    """

    def __init__(self, pos_threshold=0.5, neg_threshold=0.4):
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold

    def match(self, anchors: torch.Tensor, gt_boxes: torch.Tensor, gt_labels: torch.Tensor):
        """
        anchors: Tensor (N, 4)
        gt_boxes: Tensor (M, 4)
        gt_labels: Tensor (M,)
        Returns:
            matched_gt_boxes: Tensor (N, 4)
            matched_labels: Tensor (N,) where >=0 is class_id, -1 is background, -2 is ignore.
        """
        num_anchors = anchors.shape[0]
        device = anchors.device

        if gt_boxes.numel() == 0:
            matched_gt_boxes = torch.zeros((num_anchors, 4), device=device)
            matched_labels = torch.full((num_anchors,), -1, dtype=torch.int64, device=device)
            return matched_gt_boxes, matched_labels

        # Pairwise IoU matrix (N_anchors, M_gt)
        iou_matrix = box_iou(anchors, gt_boxes)

        # Max IoU for each anchor and corresponding GT index
        max_iou_per_anchor, best_gt_idx_per_anchor = iou_matrix.max(dim=1)

        # Initialize labels to background (-1)
        matched_labels = torch.full((num_anchors,), -1, dtype=torch.int64, device=device)

        # Ignore anchors between neg_threshold and pos_threshold (-2)
        ignore_mask = (max_iou_per_anchor >= self.neg_threshold) & (max_iou_per_anchor < self.pos_threshold)
        matched_labels[ignore_mask] = -2

        # Positive anchors (IoU >= pos_threshold)
        pos_mask = max_iou_per_anchor >= self.pos_threshold
        matched_labels[pos_mask] = gt_labels[best_gt_idx_per_anchor[pos_mask]]

        # Ensure every GT box matches at least one anchor (force max IoU match)
        max_iou_per_gt, best_anchor_idx_per_gt = iou_matrix.max(dim=0)
        for gt_idx, anchor_idx in enumerate(best_anchor_idx_per_gt):
            matched_labels[anchor_idx] = gt_labels[gt_idx]
            best_gt_idx_per_anchor[anchor_idx] = gt_idx

        matched_gt_boxes = gt_boxes[best_gt_idx_per_anchor]
        return matched_gt_boxes, matched_labels
