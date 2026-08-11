"""
Retinal fundus image analysis using the FIVES dataset.

Tasks
-----
1. Retinal vessel segmentation with U-Net / ResNet-34.
2. Four-class fundus image classification with ResNet-34.
3. Basic post-processing and exploratory vessel morphology.

Expected FIVES layout
---------------------
FIVES/
├── train/
│   ├── Original/
│   └── Ground Truth/
└── test/
    ├── Original/
    └── Ground Truth/

Example
-------
python retinal_analysis.py --data-dir /path/to/FIVES --output-dir results
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torchvision.models as models
from scipy.ndimage import convolve, label
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from skimage import morphology
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


CLASS_MAP = {"A": 0, "D": 1, "G": 2, "N": 3}
CLASS_NAMES = ["AMD", "DR", "Glaucoma", "Normal"]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FIVES retinal image analysis")
    parser.add_argument("--data-dir", type=Path, required=True, help="Root directory of the FIVES dataset")
    parser.add_argument("--output-dir", type=Path, default=Path("results"), help="Directory for checkpoints and figures")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seg-batch-size", type=int, default=8)
    parser.add_argument("--class-batch-size", type=int, default=32)
    parser.add_argument("--skip-training", action="store_true", help="Evaluate existing checkpoints without training")
    return parser.parse_args()


def parse_label_from_filename(filename: str) -> int:
    """Return the FIVES disease label encoded in filenames such as '123_A.png'."""
    letter = Path(filename).stem.split("_")[-1]
    if letter not in CLASS_MAP:
        raise ValueError(f"Could not parse FIVES class label from filename: {filename}")
    return CLASS_MAP[letter]


def list_pngs(directory: Path) -> list[str]:
    files = sorted(p.name for p in directory.glob("*.png"))
    if not files:
        raise FileNotFoundError(f"No PNG images found in {directory}")
    return files


class FIVESDatasetSeg(Dataset):
    def __init__(self, image_ids, image_dir: Path, mask_dir: Path, transform=None):
        self.image_ids = list(image_ids)
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image = cv2.imread(str(self.image_dir / f"{image_id}.png"), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(self.mask_dir / f"{image_id}.png"), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise FileNotFoundError(f"Missing image or mask for {image_id}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            sample = self.transform(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        image = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float()
        mask = torch.from_numpy(np.ascontiguousarray(mask)).unsqueeze(0).float() / 255.0
        mask = (mask > 0.5).float()
        return image, mask


class FIVESDatasetClass(Dataset):
    def __init__(self, files, labels, image_dir: Path, transform=None):
        self.files = list(files)
        self.labels = list(labels)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image = cv2.imread(str(self.image_dir / self.files[idx]), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(self.image_dir / self.files[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)["image"]
        image = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float()
        return image, int(self.labels[idx])


def make_transforms():
    seg_train = A.Compose([
        A.Resize(512, 512),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    seg_eval = A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    class_train = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    class_eval = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return seg_train, seg_eval, class_train, class_eval


def build_loaders(data_dir: Path, seg_batch_size: int, class_batch_size: int, num_workers: int, seed: int):
    train_orig = data_dir / "train" / "Original"
    train_mask = data_dir / "train" / "Ground Truth"
    test_orig = data_dir / "test" / "Original"
    test_mask = data_dir / "test" / "Ground Truth"

    for path in (train_orig, train_mask, test_orig, test_mask):
        if not path.exists():
            raise FileNotFoundError(f"Expected FIVES directory not found: {path}")

    train_files = list_pngs(train_orig)
    train_labels = [parse_label_from_filename(f) for f in train_files]
    train_files, val_files, train_labels, val_labels = train_test_split(
        train_files, train_labels, test_size=0.2, random_state=seed, stratify=train_labels
    )
    test_files = list_pngs(test_orig)
    test_labels = [parse_label_from_filename(f) for f in test_files]

    seg_train_ids = [Path(f).stem for f in train_files]
    seg_val_ids = [Path(f).stem for f in val_files]
    seg_test_ids = [Path(f).stem for f in test_files]

    seg_train_tf, seg_eval_tf, class_train_tf, class_eval_tf = make_transforms()

    seg_train_ds = FIVESDatasetSeg(seg_train_ids, train_orig, train_mask, seg_train_tf)
    seg_val_ds = FIVESDatasetSeg(seg_val_ids, train_orig, train_mask, seg_eval_tf)
    seg_test_ds = FIVESDatasetSeg(seg_test_ids, test_orig, test_mask, seg_eval_tf)

    class_train_ds = FIVESDatasetClass(train_files, train_labels, train_orig, class_train_tf)
    class_val_ds = FIVESDatasetClass(val_files, val_labels, train_orig, class_eval_tf)
    class_test_ds = FIVESDatasetClass(test_files, test_labels, test_orig, class_eval_tf)

    pin = torch.cuda.is_available()
    loaders = {
        "seg_train": DataLoader(seg_train_ds, batch_size=seg_batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin),
        "seg_val": DataLoader(seg_val_ds, batch_size=seg_batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin),
        "seg_test": DataLoader(seg_test_ds, batch_size=seg_batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin),
        "class_train": DataLoader(class_train_ds, batch_size=class_batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin),
        "class_val": DataLoader(class_val_ds, batch_size=class_batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin),
        "class_test": DataLoader(class_test_ds, batch_size=class_batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin),
    }
    return loaders, seg_test_ds


def build_models(device):
    seg_model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None,
    )
    class_model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    class_model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(class_model.fc.in_features, len(CLASS_NAMES)),
    )
    return seg_model.to(device), class_model.to(device)


class DiceBCELoss(nn.Module):
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, target):
        probs = torch.sigmoid(logits)
        intersection = (probs * target).sum(dim=(1, 2, 3))
        denominator = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return (1.0 - dice.mean()) + self.bce(logits, target)


def segmentation_counts(probs, target, threshold=0.5):
    pred = probs >= threshold
    truth = target >= 0.5
    tp = torch.logical_and(pred, truth).sum().item()
    fp = torch.logical_and(pred, ~truth).sum().item()
    fn = torch.logical_and(~pred, truth).sum().item()
    tn = torch.logical_and(~pred, ~truth).sum().item()
    return tp, fp, fn, tn


def metrics_from_counts(tp, fp, fn, tn, eps=1e-8):
    return {
        "dice": (2 * tp + eps) / (2 * tp + fp + fn + eps),
        "iou": (tp + eps) / (tp + fp + fn + eps),
        "accuracy": (tp + tn + eps) / (tp + tn + fp + fn + eps),
        "sensitivity": (tp + eps) / (tp + fn + eps),
        "specificity": (tn + eps) / (tn + fp + eps),
    }


def train_segmentation(model, train_loader, val_loader, device, checkpoint: Path):
    criterion = DiceBCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)
    best_val_loss, patience = float("inf"), 0

    for epoch in range(50):
        model.train()
        train_loss = 0.0
        for images, masks in tqdm(train_loader, desc=f"Seg {epoch + 1:02d}", leave=False):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                val_loss += criterion(model(images), masks).item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        print(f"Seg epoch {epoch + 1:02d}: train loss={train_loss:.4f}, val loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss, patience = val_loss, 0
            torch.save(model.state_dict(), checkpoint)
        else:
            patience += 1
            if patience >= 15:
                break


def classification_loss(device):
    # The Normal class receives a larger penalty because the original experiments
    # showed systematic Normal-to-Glaucoma confusion.
    return nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0, 1.0, 2.0], device=device))


def run_class_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    loss_sum, predictions, truth = 0.0, [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        if training:
            loss.backward()
            optimizer.step()
        loss_sum += loss.item()
        predictions.extend(outputs.argmax(1).detach().cpu().numpy())
        truth.extend(labels.detach().cpu().numpy())

    return loss_sum / len(loader), accuracy_score(truth, predictions)


def train_classification(model, train_loader, val_loader, device, phase1_checkpoint: Path, final_checkpoint: Path):
    criterion = classification_loss(device)

    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)
    best_acc, patience = -1.0, 0

    for epoch in range(10):
        train_loss, train_acc = run_class_epoch(model, train_loader, criterion, device, optimizer)
        with torch.no_grad():
            val_loss, val_acc = run_class_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        print(f"Cls phase 1 epoch {epoch + 1:02d}: train={train_acc:.4f}, val={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc, patience = val_acc, 0
            torch.save(model.state_dict(), phase1_checkpoint)
        else:
            patience += 1
            if patience >= 5:
                break

    model.load_state_dict(torch.load(phase1_checkpoint, map_location=device))
    for p in model.parameters():
        p.requires_grad = True

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)
    best_acc, patience = -1.0, 0

    for epoch in range(20):
        train_loss, train_acc = run_class_epoch(model, train_loader, criterion, device, optimizer)
        with torch.no_grad():
            val_loss, val_acc = run_class_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        print(f"Cls phase 2 epoch {epoch + 1:02d}: train={train_acc:.4f}, val={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc, patience = val_acc, 0
            torch.save(model.state_dict(), final_checkpoint)
        else:
            patience += 1
            if patience >= 10:
                break


def evaluate_segmentation(model, loader, device, checkpoint: Path):
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    tp = fp = fn = tn = 0
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            probs = torch.sigmoid(model(images))
            counts = segmentation_counts(probs, masks)
            tp += counts[0]; fp += counts[1]; fn += counts[2]; tn += counts[3]
    return metrics_from_counts(tp, fp, fn, tn)


def evaluate_classification(model, loader, device, checkpoint: Path, output_dir: Path):
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    preds, truth, probabilities = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(device))
            probs = torch.softmax(outputs, dim=1)
            preds.extend(outputs.argmax(1).cpu().numpy())
            truth.extend(labels.numpy())
            probabilities.extend(probs.cpu().numpy())

    preds, truth, probabilities = map(np.asarray, (preds, truth, probabilities))
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, preds, average="weighted", zero_division=0
    )
    metrics = {
        "accuracy": accuracy_score(truth, preds),
        "weighted_precision": precision,
        "weighted_recall": recall,
        "weighted_f1": f1,
    }

    cm = confusion_matrix(truth, preds)
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(cmap="Blues", values_format="d")
    plt.title("Classification confusion matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=200)
    plt.close()

    y_true = label_binarize(truth, classes=np.arange(len(CLASS_NAMES)))
    plt.figure()
    aucs = {}
    for i, name in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(y_true[:, i], probabilities[:, i])
        aucs[name] = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC={aucs[name]:.3f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("One-vs-rest ROC curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curves.png", dpi=200)
    plt.close()
    metrics["auc"] = aucs
    return metrics


def post_process_mask(mask, kernel_size=3):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)


def vessel_features(mask):
    """Return exploratory pixel-space vessel descriptors.

    These are not calibrated clinical morphometrics. Branch points are estimated
    by clustering adjacent skeleton pixels that each have at least three neighbours.
    """
    binary = mask.astype(bool)
    area_fraction = float(binary.mean())
    if not binary.any():
        return {"area_fraction": area_fraction, "skeleton_length_px": 0, "branch_point_candidates": 0}

    skeleton = morphology.skeletonize(binary)
    neighbours = convolve(
        skeleton.astype(np.uint8),
        np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8),
        mode="constant",
        cval=0,
    )
    junction_pixels = skeleton & (neighbours >= 3)
    _, n_clusters = label(junction_pixels, structure=np.ones((3, 3), dtype=np.uint8))
    return {
        "area_fraction": area_fraction,
        "skeleton_length_px": int(skeleton.sum()),
        "branch_point_candidates": int(n_clusters),
    }


def visualize_predictions(model, dataset, device, checkpoint: Path, output_dir: Path, n=3):
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    n = min(n, len(dataset))
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    axes = np.atleast_2d(axes)
    mean, std = np.array(IMAGENET_MEAN), np.array(IMAGENET_STD)

    for i in range(n):
        image, mask = dataset[i]
        with torch.no_grad():
            pred = torch.sigmoid(model(image.unsqueeze(0).to(device))).squeeze().cpu().numpy()
        pred = (pred >= 0.5).astype(np.uint8)

        display = image.permute(1, 2, 0).numpy() * std + mean
        axes[i, 0].imshow(np.clip(display, 0, 1))
        axes[i, 1].imshow(mask.squeeze().numpy(), cmap="gray")
        axes[i, 2].imshow(pred, cmap="gray")
        for j, title in enumerate(("Fundus image", "Ground truth", "Prediction")):
            axes[i, j].set_title(title)
            axes[i, j].axis("off")

    plt.tight_layout()
    plt.savefig(output_dir / "segmentation_examples.png", dpi=200)
    plt.close()


def main():
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    loaders, seg_test_dataset = build_loaders(
        args.data_dir, args.seg_batch_size, args.class_batch_size, args.num_workers, args.seed
    )
    seg_model, class_model = build_models(device)

    seg_ckpt = args.output_dir / "best_seg_unet.pth"
    cls_phase1_ckpt = args.output_dir / "best_class_resnet_phase1.pth"
    cls_ckpt = args.output_dir / "best_class_resnet.pth"

    if not args.skip_training:
        train_segmentation(seg_model, loaders["seg_train"], loaders["seg_val"], device, seg_ckpt)
        train_classification(
            class_model, loaders["class_train"], loaders["class_val"], device, cls_phase1_ckpt, cls_ckpt
        )

    seg_metrics = evaluate_segmentation(seg_model, loaders["seg_test"], device, seg_ckpt)
    cls_metrics = evaluate_classification(class_model, loaders["class_test"], device, cls_ckpt, args.output_dir)
    visualize_predictions(seg_model, seg_test_dataset, device, seg_ckpt, args.output_dir)

    with open(args.output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"segmentation": seg_metrics, "classification": cls_metrics}, f, indent=2)

    print("Segmentation:", seg_metrics)
    print("Classification:", cls_metrics)


if __name__ == "__main__":
    main()
