# src/train.py
import argparse
import os
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score

from datasets import UCF101SkeletonDataset, train_val_split
from model import TemporalMeanMLP, LSTMSkeletonClassifier


def parse_args():
    parser = argparse.ArgumentParser(description="Entrenamiento en UCF101 Skeleton (subset)")

    parser.add_argument("--pkl_path", type=str, default="data/ucf101_2d.pkl")
    parser.add_argument("--split_name", type=str, default="train",
                        help="Nombre del split dentro del pkl (ej: 'train', 'train_joint').")
    parser.add_argument("--selected_labels", type=int, nargs="+", default=None,
                        help="IDs de clase a usar, ej: --selected_labels 0 1 2 3 4")
    parser.add_argument("--max_frames", type=int, default=64)

    parser.add_argument("--model_type", type=str, default="lstm", choices=["baseline", "lstm"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--output_dir", type=str, default="checkpoints")

    return parser.parse_args()


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    all_losses = []
    all_preds = []
    all_labels = []

    for x, y in tqdm(loader, desc="Train", leave=False):
        x = x.to(device)  # (B, T, D)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        all_losses.append(loss.item())
        preds = logits.argmax(dim=1).detach().cpu()
        all_preds.extend(preds.numpy().tolist())
        all_labels.extend(y.detach().cpu().numpy().tolist())

    acc = accuracy_score(all_labels, all_preds)
    return sum(all_losses) / len(all_losses), acc


def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    all_losses = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Val", leave=False):
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            all_losses.append(loss.item())
            preds = logits.argmax(dim=1).detach().cpu()
            all_preds.extend(preds.numpy().tolist())
            all_labels.extend(y.detach().cpu().numpy().tolist())

    acc = accuracy_score(all_labels, all_preds)
    return sum(all_losses) / len(all_losses), acc


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Dataset completo (solo split_name y selected_labels)
    base_dataset = UCF101SkeletonDataset(
        pkl_path=args.pkl_path,
        split_name=args.split_name,
        selected_labels=args.selected_labels,
        max_frames=args.max_frames,
    )

    # Obtenemos input_dim = 2 * num_joints
    input_dim = 2 * base_dataset.num_joints

    # Creamos split train/val
    train_ds, val_ds = train_val_split(base_dataset, val_ratio=args.val_ratio)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Mapeo de labels seleccionados a [0..C-1] (para que sea compacto)
    if args.selected_labels is not None:
        unique_labels: List[int] = sorted(list(set(args.selected_labels)))
    else:
        # Tomamos todos los labels presentes en base_dataset.samples
        unique_labels = sorted({int(ann["label"]) for ann in base_dataset.samples})

    label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}

    # Reempaquetamos para que el modelo trabaje con labels 0..C-1
    # Simple truco: envolvemos los dataloaders con un pequeño wrapper…
    def relabel_batch(batch):
        x, y = batch
        # mapeamos y (clase original) a índice local
        y_mapped = torch.tensor([label_to_idx[int(yi)] for yi in y], dtype=torch.long)
        return x, y_mapped

    # Wrapper simple: aplicamos relabel_batch en el loop (abajo).

    num_classes = len(unique_labels)

    # Definimos modelo
    if args.model_type == "baseline":
        model = TemporalMeanMLP(input_dim=input_dim, num_classes=num_classes)
    else:
        model = LSTMSkeletonClassifier(input_dim=input_dim, num_classes=num_classes)

    device = args.device
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    best_path = os.path.join(args.output_dir, f"best_{args.model_type}.pt")

    for epoch in range(1, args.num_epochs + 1):
        print(f"\nEpoch {epoch}/{args.num_epochs}")

        # Train
        train_batches = []
        for batch in train_loader:
            train_batches.append(relabel_batch(batch))
        train_loss, train_acc = train_one_epoch(model, train_batches, criterion, optimizer, device)

        # Val
        val_batches = []
        for batch in val_loader:
            val_batches.append(relabel_batch(batch))
        val_loss, val_acc = eval_one_epoch(model, val_batches, criterion, device)

        print(
            f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f} | "
            f"Val loss: {val_loss:.4f} | Val acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "label_to_idx": label_to_idx,
                    "input_dim": input_dim,
                    "num_classes": num_classes,
                    "model_type": args.model_type,
                    "max_frames": args.max_frames,
                },
                best_path,
            )
            print(f"Nuevo mejor modelo guardado en {best_path} (val_acc={best_val_acc:.4f})")

    print(f"\nMejor val_acc: {best_val_acc:.4f}")
    print(f"Modelo final guardado en: {best_path}")


if __name__ == "__main__":
    main()