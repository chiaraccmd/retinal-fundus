# 👁️ Retinal Image Analysis with Deep Learning

*A PyTorch pipeline for retinal vessel segmentation, fundus image classification, and exploratory vascular morphology using the FIVES dataset.*

---

## 🚀 Overview

This project analyses retinal fundus images from the **FIVES** dataset using deep learning and classical image processing.

The pipeline addresses two complementary tasks:

- **Retinal vessel segmentation** using a U-Net with a ResNet-34 encoder.
- **Four-class fundus image classification** using a pretrained ResNet-34.

The segmentation output is also used for simple post-processing and exploratory vessel morphology measurements.

The FIVES dataset contains 800 high-resolution fundus photographs with pixel-wise vessel annotations and four disease categories: **AMD, diabetic retinopathy (DR), glaucoma, and normal**.

---

## 🧠 Pipeline

```text
FIVES fundus images
        │
        ├───────────────────────┐
        ↓                       ↓
Vessel segmentation       Disease classification
U-Net + ResNet-34             ResNet-34
        ↓                       ↓
Dice / IoU               Accuracy / F1
Sensitivity / Specificity     ROC / AUC
        ↓
Morphological closing
        ↓
Skeletonization
        ↓
Exploratory vessel features
```

---

## 📁 Repository Structure

```text
retinal-image-analysis/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   └── retinal_analysis.py
└── results/
    └── .gitkeep
```

The FIVES dataset and trained model weights are not distributed with this repository.

---

## 📊 Dataset

The project uses the **FIVES: A Fundus Image Dataset for Artificial Intelligence based Vessel Segmentation** dataset.

FIVES contains:

- 800 colour fundus photographs at 2048 × 2048 pixels;
- pixel-wise retinal vessel annotations;
- 200 images for each of four categories:
  - AMD;
  - diabetic retinopathy;
  - glaucoma;
  - normal.

The official training set is divided into training and validation subsets using a stratified 80/20 split. The official test set remains separate for final evaluation.

Expected directory structure:

```text
FIVES/
├── train/
│   ├── Original/
│   └── Ground Truth/
└── test/
    ├── Original/
    └── Ground Truth/
```

---

## 🩸 Vessel Segmentation

Retinal vessels are segmented using **U-Net** with an ImageNet-pretrained **ResNet-34 encoder**.

Images are resized to 512 × 512 pixels. Training augmentation includes horizontal flips, small rotations, and mild brightness/contrast variation.

The model is trained using a combined **Dice + Binary Cross-Entropy loss**:

```text
segmentation logits
        ↓
sigmoid probabilities
        ↓
Dice loss + BCEWithLogitsLoss
```

The combined objective is useful for vessel segmentation because vessel pixels occupy a relatively small fraction of the retinal image.

### Segmentation metrics

The test pipeline reports:

- Dice coefficient;
- Intersection over Union (IoU);
- pixel accuracy;
- sensitivity;
- specificity.

In the original project experiments, the segmentation model achieved a **test Dice coefficient of 0.8856**.

---

## 🔬 Fundus Classification

A pretrained **ResNet-34** is adapted for four-class classification:

```text
fundus image
     ↓
ResNet-34
     ↓
dropout
     ↓
4-class classifier
     ↓
AMD / DR / Glaucoma / Normal
```

Images are resized to 224 × 224 pixels and normalized using ImageNet statistics.

Training uses two stages:

1. train the classification head while the backbone is frozen;
2. unfreeze the network and fine-tune all layers with a lower learning rate.

The classification loss assigns a larger weight to the Normal class because the original experiments showed systematic confusion between Normal and Glaucoma images.

### Classification metrics

Evaluation includes:

- accuracy;
- weighted precision;
- weighted recall;
- weighted F1-score;
- confusion matrix;
- one-vs-rest ROC curves and AUC.

The original project obtained **81.0% test accuracy** and a **weighted F1-score of 0.808**.

---

## 🔍 Post-Processing & Vessel Morphology

Predicted vessel masks can be refined using morphological closing to fill small gaps and smooth local discontinuities.

The cleaned binary mask is then skeletonized to obtain simple pixel-space descriptors:

- vessel area fraction;
- skeleton length;
- branch-point candidate count.

These quantities are intended as **exploratory image descriptors**, not calibrated clinical biomarkers. In particular, branch points are estimated from connected clusters of skeleton pixels with at least three neighbours.

---

## 🛠️ Installation

Create a Python environment and install the dependencies:

```bash
pip install -r requirements.txt
```

The main dependencies are PyTorch, torchvision, segmentation-models-pytorch, Albumentations, OpenCV, scikit-learn, scikit-image, SciPy, Matplotlib, and tqdm.

---

## ▶️ Usage

Train both models and evaluate them:

```bash
python src/retinal_analysis.py \
    --data-dir /path/to/FIVES \
    --output-dir results
```

To evaluate previously trained checkpoints without retraining:

```bash
python src/retinal_analysis.py \
    --data-dir /path/to/FIVES \
    --output-dir results \
    --skip-training
```

Outputs include model checkpoints, segmentation examples, a classification confusion matrix, ROC curves, and a JSON file containing the final metrics.

---

## 🧪 Reproducibility

The script uses a fixed random seed by default and performs a stratified training/validation split.

```bash
python src/retinal_analysis.py \
    --data-dir /path/to/FIVES \
    --seed 42
```

GPU acceleration is used automatically when CUDA is available.

Exact numerical results can still vary slightly across hardware, CUDA, PyTorch, and dependency versions.

---

## 📈 Original Results

| Task | Metric | Result |
| --- | --- | ---: |
| Vessel segmentation | Dice | **0.8856** |
| Fundus classification | Accuracy | **81.0%** |
| Fundus classification | Weighted F1 | **0.808** |

Per-class classification performance from the original experiment:

| Class | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| AMD | 0.896 | 0.860 | 0.878 |
| DR | 0.808 | 0.840 | 0.824 |
| Glaucoma | 0.697 | 0.920 | 0.793 |
| Normal | 0.912 | 0.620 | 0.738 |

The largest classification limitation was the remaining confusion between **Glaucoma and Normal** images.

---

## 🔄 Possible Extensions

- Multi-task learning with a shared encoder for segmentation and classification.
- Higher-resolution segmentation with memory-efficient training.
- Test-time augmentation.
- More rigorous retinal vascular morphometrics, including tortuosity and vessel calibre.
- Analysis of performance as a function of FIVES image-quality labels.
- Cross-dataset validation on additional retinal fundus datasets.

---

## 📚 References

- Jin, K. et al. *FIVES: A Fundus Image Dataset for Artificial Intelligence based Vessel Segmentation*. Scientific Data, 2022.
- He, K. et al. *Deep Residual Learning for Image Recognition*. CVPR, 2016.
- Milletari, F. et al. *V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation*. 3DV, 2016.

---

*Developed as a biomedical computer-vision project for retinal vessel segmentation and fundus image classification.*
