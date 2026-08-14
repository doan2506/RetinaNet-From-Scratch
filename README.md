# RetinaNet From Scratch - Đồ Án Cuối Kỳ Object Detection

Dự án cài đặt mô hình phát hiện đối tượng **RetinaNet** từ đầu bằng PyTorch (From Scratch), huấn luyện và đánh giá trên bộ dữ liệu 5 lớp đối tượng: `person`, `car`, `dog`, `cat`, `chair`.

---

## 1. Cấu Trúc Dự Án

```text
.
├── public/                      # Bộ dữ liệu và tools đánh giá mẫu
│   ├── classes.json
│   ├── train/images/
│   ├── val/images/
│   ├── annotations/
│   │   ├── train.json
│   │   └── val.json
│   └── tools/
│       └── evaluate_predictions.py
├── models/                      # Mã nguồn kiến trúc RetinaNet & Checkpoints
│   ├── backbone.py              # ResNet Feature Extractor (C3, C4, C5)
│   ├── fpn.py                   # Feature Pyramid Network (P3→P7), P6 từ C5
│   ├── anchor.py                # Anchor Generator (9 anchors/pixel) & Matcher
│   ├── head.py                  # Classification & BBox Regression Subnets
│   ├── losses.py                # Focal Loss & Smooth L1 BBox Loss
│   └── retinanet.py             # Mô hình RetinaNet tổng hợp
├── utils/                       # Công cụ hỗ trợ
│   ├── dataset.py               # JSON Dataset loader & Multi-scale Collate
│   ├── augmentations.py         # Multi-scale, Expand+Crop, Flip, Color Jitter
│   ├── box_utils.py             # IoU, BBox Encode/Decode
│   └── nms.py                   # Per-class NMS viết bằng PyTorch thuần
├── train.py                     # Script huấn luyện (backbone freeze, warmup, mAP tracking)
├── predict.py                   # Script suy luận (xuất predictions.json)
├── requirements.txt
└── README.md
```

---

## 2. Cài Đặt Môi Trường

Khuyến nghị sử dụng Python `3.9+` và môi trường ảo `conda` hoặc `venv`:

```bash
conda create -n retinanet python=3.9 -y
conda activate retinanet
pip install -r requirements.txt
```

---

## 3. Huấn Luyện Mô Hình (`train.py`)

Chạy lệnh huấn luyện chuẩn:

```bash
python train.py \
  --train_data ./public/annotations/train.json \
  --val_data ./public/annotations/val.json \
  --image_dir ./public/train/images \
  --val_image_dir ./public/val/images \
  --checkpoint_dir ./models/
```

### Các tham số tùy chỉnh:
| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `--epochs` | 40 | Số epoch huấn luyện |
| `--batch_size` | 8 | Batch size |
| `--lr` | 1e-4 | Learning rate |
| `--backbone` | resnet50 | Backbone (resnet34/resnet50) |
| `--img_size` | 640 | Kích thước ảnh mặc định |
| `--freeze_backbone_epochs` | 2 | Số epoch freeze backbone đầu |
| `--warmup_iters` | 500 | Số iteration LR warmup |
| `--grad_clip` | 1.0 | Max gradient norm |

### Các kỹ thuật tối ưu được tích hợp:
- **Backbone Freezing**: Freeze ResNet 2 epoch đầu để ổn định head trước khi fine-tune.
- **LR Warmup + Cosine Annealing**: Linear warmup 500 iterations, rồi cosine decay.
- **Gradient Clipping**: Norm clip = 1.0, tránh exploding gradients.
- **Multi-scale Training**: Random resize ảnh trong {480...800} mỗi sample.
- **SSD-style Expand+Crop**: Random expand rồi crop, giữ ít nhất 1 GT box.
- **mAP-based Checkpoint Selection**: Nếu có `evaluate_predictions.py`, lưu model theo mAP@0.5 cao nhất thay vì val loss thấp nhất.

> **Lưu ý**: Trọng số tốt nhất tự động lưu tại `./models/best.pth`.

---

## 4. Suy Luận (`predict.py`)

```bash
python predict.py \
  --image_dir /path/to/images \
  --output predictions.json
```

Kết quả xuất ra `predictions.json` đúng định dạng:
```json
[
  {
    "image_id": "img_7fd91a4c2e30.jpg",
    "boxes": [
      {
        "class": "person",
        "confidence": 0.91,
        "bbox": [48.0, 72.0, 210.0, 356.0]
      }
    ]
  }
]
```

---

## 5. Tự Đánh Giá

```bash
python public/tools/evaluate_predictions.py \
  --ground_truth public/annotations/val.json \
  --predictions val_predictions.json \
  --output val_score.json
```

---

## 6. Vị Trí Trọng Số Mô Hình

- `./models/best.pth` — Checkpoint tốt nhất (tự động lưu bởi `train.py`).
