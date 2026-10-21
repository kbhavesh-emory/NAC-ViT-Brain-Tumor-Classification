# NAC-ViT: Neuronal Attention Circuit Vision Transformer for Brain Tumor Classification and Detection

## Overview

NAC-ViT (Neuronal Attention Circuit Vision Transformer) is a biologically-inspired deep learning architecture for brain tumor MRI classification and detection. The project combines:

- **NAC-ViT**: A Vision Transformer with ODE-based attention dynamics for brain tumor classification
- **YOLO11m**: Tumor detection, trained separately from the classifier
- **Two-stage design**: Detection, then classification. The two models were not evaluated as one pipeline

## Project Structure

```
NAC-ViT/
├── train_nac_vit.py              # NAC-ViT classification training
├── train_yolo_brain_tumor.py     # YOLO detection training
├── NAC-ViT-view.ipynb            # Classification result visualizations
├── datasets/                     # Training datasets (not in git)
│   ├── brain-tumor/              # YOLO detection dataset
│   └── Brain-Tumor-Classification-DataSet-master/
├── runs/
│   ├── classify/nac_vit_pooled/  # Unfrozen 80/10/10 run (96.36%)
│   ├── classify/nac_vit/         # Official Testing-folder run
│   └── detect/                   # YOLO training results
```

## Installation

### Requirements
- Python 3.8+
- PyTorch 2.0+
- CUDA-compatible GPU
- Ultralytics YOLO (detection training only)

### Setup
```bash
python -m venv .venv
source .venv/bin/activate

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics pandas matplotlib jupyter python-docx
```

## Datasets

### Classification Dataset
- **Source**: [Brain Tumor Classification (MRI)](https://github.com/sartajbhuvaji/Brain-Tumor-Classification-DataSet)
- **Classes**: Glioma, Meningioma, No Tumor, Pituitary
- **Images**: 3,264
- **Split used for the 96.36% result**: pooled 80/10/10 (2,609 train / 325 val / 330 test)

The classification script downloads this dataset when `--download` is set.

### Detection Dataset
- **Source**: Ultralytics Brain Tumor Detection Dataset
- **Classes**: Negative (no tumor), Positive (tumor present)
- **Images**: 893 training, 223 validation
- **Purpose**: YOLO tumor localization

## Usage

### NAC-ViT Classification Training

Unfrozen ViT-B/16 fine-tune on the pooled 80/10/10 split:

```bash
python train_nac_vit.py --download --mode full --epochs 40 --split pooled
```

One-epoch pipeline check:

```bash
python train_nac_vit.py --download --mode test
```

Keep the dataset's Testing folder as the test set:

```bash
python train_nac_vit.py --split official --mode full --epochs 40 --output runs/classify/nac_vit
```

Evaluate a saved checkpoint:

```bash
python train_nac_vit.py --eval-only \
  --checkpoint runs/classify/nac_vit_pooled/best.pt \
  --output runs/classify/nac_vit_pooled
```

The unfrozen run trains all 88.1M parameters. The backbone learning rate is `1e-5` and the NAC blocks plus classification head use `1e-4`. Training stops early when validation accuracy does not improve for 8 epochs.

### YOLO Tumor Detection Training

The recorded detection run used YOLO11m for 300 epochs:

```bash
python train_yolo_brain_tumor.py --mode full --epochs 300 --imgsz 640 --model m
```

Evaluation only: `python train_yolo_brain_tumor.py --eval-only`.

### Results Visualization

```bash
jupyter notebook NAC-ViT-view.ipynb
```

The viewer loads `runs/classify/nac_vit_pooled/` and writes figures to `runs/classify/nac_vit_pooled/figures/`. It includes training curves, per-class metrics, protocol comparison, confusion matrices, ROC curves, confidence histograms, prediction examples, split balance, and NAC dynamics.

## Model Architecture

**NAC-ViT components in this training script:**
- Backbone: ViT-B/16 pretrained on ImageNet-1K (torchvision), unfrozen during the 96.36% run
- 4 NAC blocks with ODE attention: `dA/dt = φ·A(t)·(1-A(t)) - ω·A(t)`
- Classification head: linear layer, 4 classes
- Trainable parameters: 88.1M unfrozen, about 2.3M when the backbone is frozen

**Biological parameters stored in the checkpoint:**
- φ (gate): logistic term, one positive value per attention head
- ω (leak): one positive value per attention head
- τ = 1/ω

## Results

### NAC-ViT Classification (Unfrozen, pooled 80/10/10)

Trained on an NVIDIA RTX A5000, batch size 16. Best validation epoch was 9. Early stopping ended the run at epoch 17. Wall time was 6.5 minutes.

| Split | Accuracy | Macro F1 | AUC |
|---|---:|---:|---:|
| Validation | 97.85% | 0.981 | 0.992 |
| Test | **96.36%** | **0.964** | **0.998** |

Test set, 330 images:

| Class | Precision | Recall | F1 | AUC | Support |
|---|---:|---:|---:|---:|---:|
| Glioma | 0.967 | 0.936 | 0.951 | 0.995 | 94 |
| Meningioma | 0.949 | 0.979 | 0.964 | 0.999 | 95 |
| No Tumor | 0.926 | 1.000 | 0.962 | 1.000 | 50 |
| Pituitary | 1.000 | 0.956 | 0.978 | 1.000 | 91 |

Checkpoint: `runs/classify/nac_vit_pooled/best.pt`

### NAC-ViT on the dataset Testing folder

The same unfrozen recipe, with the dataset's Testing folder held out, reached **78.43%** test accuracy (309/394; macro F1 0.750, AUC 0.924). Validation accuracy on a 10% slice of the Training folder was 99.30%. Glioma recall on that Testing folder was 0.240.

### YOLO11m Detection

Trained for 300 epochs at 640 × 640. The epoch with the highest mAP@0.50 was epoch 27. Later epochs did not improve that score.

| Metric | Value |
|---|---:|
| Precision | 0.417 |
| Recall | 0.795 |
| mAP@0.50 | 0.477 |
| mAP@0.50:0.95 | 0.274 |

Weights: `runs/detect/brain_tumor_yolo/weights/best.pt`

## Two-Stage Clinical Pipeline

1. **Stage 1**: YOLO11m localizes the tumor
2. **Stage 2**: NAC-ViT classifies an image into 4 classes

No experiment in this project passes detector boxes into NAC-ViT. The two results above are separate measurements.

## Hardware Requirements

- **GPU**: NVIDIA GPU with CUDA. The 96.36% run used an RTX A5000 (22 GB)
- **Memory**: 16 GB+ GPU memory for unfrozen ViT-B/16 at batch size 16. Use `--batch-size 8` on a 10 GB card
- **Storage**: 10 GB+ for datasets, logs, and checkpoints

## Troubleshooting

### CUDA Out of Memory
- Classification: `python train_nac_vit.py --batch-size 8 --split pooled`
- Detection: `--imgsz 512` or `--model n`

### Dataset Download Issues
- Pass `--download` on the classification script, or `--data /path/to/dataset`
- Class folders should be named glioma, meningioma, no_tumor, and pituitary

## License

This project is for research purposes. Please cite the paper if you use this code.

## Citation

```bibtex
@article{kataria2025nacvit,
  title={NAC-ViT: Neuronal Attention Circuit Vision Transformer for Brain Tumor Detection and Classification in MRI},
  author={Kataria, Bhavesh},
}
```

## Contact

- **Author**: Dr. Bhavesh Kataria
- **GitHub**: https://github.com/kbhavesh-emory/NAC-ViT-Brain-Tumor-Classification
 

## Acknowledgments

- Brain Tumor Classification dataset (Sartaj Bhuvaji)
- Ultralytics Brain Tumor Detection Dataset
- Vision Transformer (Dosovitskiy et al., 2021)
- Neural ODEs (Chen et al., 2018)
