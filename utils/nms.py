import torch
from utils.box_utils import box_area


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    """
    Pure PyTorch Non-Maximum Suppression (NMS).
    boxes: Tensor (N, 4) in [xmin, ymin, xmax, ymax]
    scores: Tensor (N,) confidence scores
    iou_threshold: Float threshold for IoU suppression
    Returns: Tensor of kept indices
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    order = scores.argsort(descending=True)

    keep = []
    while order.numel() > 0:
        if order.numel() == 1:
            i = order.item()
            keep.append(i)
            break

        i = order[0].item()
        keep.append(i)

        # Compute IoU of box i with remaining boxes in order[1:]
        xx1 = torch.maximum(x1[i], x1[order[1:]])
        yy1 = torch.maximum(y1[i], y1[order[1:]])
        xx2 = torch.minimum(x2[i], x2[order[1:]])
        yy2 = torch.minimum(y2[i], y2[order[1:]])

        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h

        rem_areas = areas[order[1:]]
        union = areas[i] + rem_areas - inter
        union = torch.clamp(union, min=1e-6)

        iou = inter / union
        mask = iou <= iou_threshold

        # Keep remaining boxes where IoU <= threshold
        order = order[1:][mask]

    return torch.tensor(keep, dtype=torch.int64, device=boxes.device)


def batched_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    iou_threshold: float
) -> torch.Tensor:
    """
    Per-class Non-Maximum Suppression.
    boxes: Tensor (N, 4)
    scores: Tensor (N,)
    labels: Tensor (N,)
    iou_threshold: Float
    Returns: Tensor of kept indices
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)

    # Offset boxes per class so boxes of different classes don't suppress each other
    max_coordinate = boxes.max()
    offsets = labels.to(boxes.dtype) * (max_coordinate + 1.0)
    boxes_for_nms = boxes + offsets[:, None]

    return nms(boxes_for_nms, scores, iou_threshold)
