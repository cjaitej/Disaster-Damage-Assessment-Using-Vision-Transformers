# Disaster Damage Analysis Using Transformer Architecture

> Semantic segmentation of UAV aerial imagery for disaster response using a Hybrid Attention Transformer (HAT) — achieving **77.35% mIoU** at 384×384 resolution.

---

## 📖 Overview

This project presents a transformer-based segmentation model for post-disaster scene understanding using the [RescueNet](https://github.com/BinaLab/RescueNet-A-High-Resolution-Post-Disaster-UAV-Dataset-for-Semantic-Segmentation) dataset. The model is designed to operate at a **reduced input resolution (384×384)** while maintaining competitive performance, making it practical for resource-constrained deployment in real-world disaster response.

The architecture combines:
- A **Hybrid Attention Transformer (HAT)** encoder
- An **Attention-based Feature Fusion** module (FPN-style)
- A **U-Net style decoder** with gated skip connections
- A **Boundary Refinement Module** for sharper edge localization

---

## 🏗️ Architecture

![Overall Architecture](https://raw.githubusercontent.com/cjaitej/Disaster-Damage-Assessment-Using-Vision-Transformers/main/Arch3.png)

![HAT Block](https://raw.githubusercontent.com/cjaitej/Disaster-Damage-Assessment-Using-Vision-Transformers/main/Arch2.png)

![Overlapping Cross-Attention](https://raw.githubusercontent.com/cjaitej/Disaster-Damage-Assessment-Using-Vision-Transformers/main/Arch1.png)

### Key Components

| Module | Description |
|---|---|
| **HAT Block** | Parallel local (OCA) + global (GSA) attention with learnable gate fusion |
| **Overlapping Cross-Attention (OCA)** | Window attention with extended key-value region to avoid boundary artefacts |
| **Global Sparse Attention (GSA)** | Stride-sampled tokens for O(N²/s²) long-range context |
| **Feature Fusion** | Top-down FPN pathway with channel-wise attention |
| **Attention Gates** | Gated skip connections to suppress noise in the decoder |
| **Boundary Refinement** | Dilated convolutions + spatial/channel attention for edge sharpness |

---

## 📊 Results

### Segmentation Performance (IoU %)

| Method | Resolution | Vehicle | Water | Tree | mIoU |
|---|---|---|---|---|---|
| Segmenter | 713×713 | 66.69 | 97.46 | 98.58 | **85.78** |
| **Ours (HAT)** | **384×384** | **75.27** | 90.39 | 96.49 | **77.35** |
| Segmenter | 384×384 | 37.60 | 85.99 | 94.47 | 63.84 |

> Our model at 384×384 outperforms the Segmenter baseline at the **same resolution by +13.51% mIoU**, and achieves **better vehicle IoU** than even the higher-resolution Segmenter (75.27 vs 66.69).

### Computational Efficiency

| Model | Params | Input Size | Inference Time |
|---|---|---|---|
| Segmenter | 26.4M | 713×713 | 67.51 ms |
| **HAT (Ours)** | 30.7M | 384×384 | **44.00 ms** |

---

## 🗂️ Dataset

**RescueNet** — High-resolution UAV aerial imagery for disaster damage assessment.

- **Train / Val / Test:** 3,595 / 449 / 450 image-mask pairs
- **Original resolution:** 3000×4000 px (resized to 384×384 for training)
- **11 semantic classes:**

| Class | Class |
|---|---|
| Background | Vehicle |
| Water | Road (Clear) |
| Building (No Damage) | Road (Blocked) |
| Building (Minor Damage) | Tree |
| Building (Major Damage) | Pool |
| Building (Total Destruction) | |

---

## ⚙️ Training Details

| Setting | Value |
|---|---|
| Framework | PyTorch |
| GPU | NVIDIA RTX A6000 (48GB) |
| Input Resolution | 384×384 |
| Batch Size | 24 |
| Epochs | 500 |
| Optimizer | AdamW (weight decay: 0.01) |
| Learning Rate | 3×10⁻⁴ → 1×10⁻⁷ (cosine annealing) |
| Metric | Mean IoU (mIoU) |

### Loss Function

```
L_total = L_main + 0.5 * L_aux + 0.3 * L_bce
L_seg   = 0.4 * L_focal + 0.6 * L_lovasz
```

---



## 📄 License

This project is released for academic and research purposes. Please cite appropriately if you use this work.

---

## 🙏 Acknowledgements

- [RescueNet Dataset](https://github.com/BinaLab/RescueNet-A-High-Resolution-Post-Disaster-UAV-Dataset-for-Semantic-Segmentation) by Rahnemoonfar et al.
- [Swin Transformer](https://github.com/microsoft/Swin-Transformer)
- [SegFormer](https://github.com/NVlabs/SegFormer)
- [Attention UNet](https://arxiv.org/abs/1804.03999)
