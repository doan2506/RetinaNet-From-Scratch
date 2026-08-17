# RetinaNet From Scratch — Object Detection Final Project (Indoor5)

Mô hình phát hiện đối tượng **RetinaNet** được cài đặt từ đầu bằng PyTorch (From Scratch), huấn luyện và đánh giá trên bộ dữ liệu 5 lớp đồ vật trong nhà (`indoor5`):
- `bottle`
- `cup`
- `chair`
- `laptop`
- `backpack`

---

## 1. Cấu Trúc Mã Nguồn

```text
<my_submission>/
├── models/
│   ├── backbone.py              # ResNet Feature Extractor (C3, C4, C5)
│   ├── fpn.py                   # Feature Pyramid Network (P3→P7)
│   ├── anchor.py                # Multi-level Anchor Generator & Matcher
│   ├── head.py                  # Classification & Regression Subnets
│   ├── losses.py                # Sigmoid Focal Loss & Smooth L1 Loss
│   └── retinanet.py             # Kiến trúc RetinaNet hoàn chỉnh
├── utils/
│   ├── dataset.py               # JSON Dataset loader & Multi-scale padding
│   ├── augmentations.py         # Multi-scale, Expand+Crop, Flip, Color Jitter
│   ├── box_utils.py             # IoU, BBox Transform & Decode
│   └── nms.py                   # Per-class NMS (Non-Maximum Suppression)
├── train.py                     # Script huấn luyện (DDP, AMP, TensorBoard, Warmup)
├── predict.py                   # Script suy luận (Auto-download weights, TTA)
├── requirements.txt             # Danh sách thư viện phụ thuộc
└── README.md                    # Tài liệu hướng dẫn sử dụng
```

---

## 2. Cài Đặt Môi Trường

Mô hình tương thích hoàn toàn với môi trường chuẩn Docker của cuộc thi (`object-detection-exam:2026` với PyTorch 2.7.1, CUDA 12.6, cuDNN 9).

Cài đặt trên máy cá nhân:
```bash
pip install -r requirements.txt
```

---

## 3. Hướng Dẫn Huấn Luyện Mô Hình (`train.py`)

### 3.1. Lệnh Huấn Luyện Chuẩn (Theo Quy Định)

```bash
python train.py \
  --train_data ./public/annotations/train.json \
  --val_data ./public/annotations/val.json \
  --image_dir ./public/train/images \
  --val_image_dir ./public/val/images \
  --checkpoint_dir ./models/
```

### 3.2. Huấn Luyện Phân Tán Đa GPU (Multi-GPU DDP trên Kaggle 2x T4):

```bash
torchrun --nproc_per_node=2 train.py \
  --train_data ./public/annotations/train.json \
  --val_data ./public/annotations/val.json \
  --image_dir ./public/train/images \
  --val_image_dir ./public/val/images \
  --checkpoint_dir ./models/ \
  --epochs 40 \
  --batch_size 16 \
  --lr 1e-4 \
  --img_size 640 \
  --backbone resnet50
```

### 3.3. Các Kỹ Thuật Tối Ưu Huấn Luyện:
* **Differential Learning Rates**: Backbone fine-tuning với tốc độ học nhỏ hơn 10 lần (`LR = 1e-5`) so với Head (`LR = 1e-4`).
* **Linear Warmup + Cosine Annealing**: Warmup mượt mà 500 bước đầu và cosine decay dần về 0.
* **Mixed Precision (AMP FP16)**: Tối ưu bộ nhớ VRAM trên GPU Tensor Cores.
* **Multi-scale & Data Augmentation**: Tự động co giãn kích thước {480...800}, lật ngang, color jitter và SSD-style expand-crop.
* **TensorBoard Logging**: Tự động lưu biểu đồ huấn luyện vào thư mục `./runs/`.

---

## 4. Hướng Dẫn Suy Luận (`predict.py`)

### 4.1. Lệnh Suy Luận Chuẩn (Theo Quy Định Chấm Bài)

```bash
python predict.py \
  --image_dir /path/to/images \
  --output predictions.json
```

Để tối ưu độ chính xác và đạt mAP cao nhất, có thể bật **Test-Time Augmentation (TTA)**:
```bash
python predict.py \
  --image_dir /path/to/images \
  --output predictions.json \
  --use_tta
```

### 4.2. Định Dạng Kết Quả Xuất Ra (`predictions.json`):

```json
[
  {
    "image_id": "img_7fd91a4c2e30.jpg",
    "boxes": [
      {
        "class": "chair",
        "confidence": 0.91,
        "bbox": [48.0, 72.0, 210.0, 356.0]
      }
    ]
  }
]
```

---

## 5. Vị Trí Trọng Số & Cơ Chế Tự Động Tải (Auto-Download Weights)

### 5.1. Vị Trí Lưu Trọng Số:
* Khi chạy huấn luyện (`train.py`), checkpoint tốt nhất sẽ tự động được lưu tại: `./models/best.pth`.
* Theo đúng quy chế chấm thi, **tệp `.pth` không được nén kèm trong bài nộp**.

### 5.2. URL & Cơ Chế Tự Động Tải:
* **URL tải trọng số mặc định:**
  `https://github.com/doan2506/RetinaNet-From-Scratch/releases/download/v1.0/best.pth`
* **Cơ chế hoạt động:**
  Khi hệ thống chấm bài gọi `predict.py`, hàm `load_model()` sẽ kiểm tra sự tồn tại của `./models/best.pth`. Nếu chưa có, hàm `download_weights()` dùng module chuẩn `urllib.request` của Python để tự động tải file trọng số về thư mục `./models/best.pth` trước khi bắt đầu suy luận.

---

## 6. Đánh Giá Kết Quả Kiểm Định (Validation mAP@0.5)

Chạy công cụ đánh giá đi kèm để kiểm tra mAP@0.5 trên tập validation:

```bash
python public/tools/evaluate_predictions.py \
  --ground_truth public/annotations/val.json \
  --predictions predictions.json \
  --output val_score.json
```
