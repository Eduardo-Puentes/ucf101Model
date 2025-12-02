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
    parser.add_argument("--split_name", type=str, default="train1",
                        help="Nombre del split dentro del pkl (ej: 'train1', 'train2', 'train3').")
    parser.add_argument("--selected_labels", type=int, nargs="+", default=None,
                        help="IDs de clase a usar, ej: --selected_labels 0 1 2 3 4")
    parser.add_argument("--max_frames", type=int, default=64)

    parser.add_argument("--model_type", type=str, default="lstm", choices=["baseline", "lstm"])
    parser.add_argument("--baseline_hidden_dim", type=int, default=128)
    parser.add_argument("--lstm_hidden_dim", type=int, default=256)
    parser.add_argument("--lstm_layers", type=int, default=2)
    parser.add_argument("--bidirectional", action="store_true",
                        help="Activa LSTM bidireccional para capturar contexto futuro/pasado.")
    parser.add_argument("--dropout", type=float, default=0.3)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="0 desactiva el clip de gradiente.")
    parser.add_argument("--scheduler_step", type=int, default=0,
                        help="Si >0, StepLR cada N epochs.")
    parser.add_argument("--scheduler_gamma", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--output_dir", type=str, default="checkpoints")

    return parser.parse_args()


def _map_labels(y_tensor, label_to_idx, device):
    mapped = [label_to_idx[int(yi)] for yi in y_tensor.detach().cpu().numpy().tolist()]
    return torch.tensor(mapped, device=device, dtype=torch.long)


def train_one_epoch(model, loader, criterion, optimizer, device, label_to_idx, grad_clip: float = 0.0):
    model.train()
    all_losses = []
    all_preds = []
    all_labels = []

    for x, y, lengths in tqdm(loader, desc="Train", leave=False):
        x = x.to(device)
        y_mapped = _map_labels(y, label_to_idx, device)

        optimizer.zero_grad()
        logits = model(x, lengths)
        loss = criterion(logits, y_mapped)
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        all_losses.append(loss.item())
        preds = logits.argmax(dim=1).detach().cpu().numpy().tolist()
        all_preds.extend(preds)
        all_labels.extend(y_mapped.detach().cpu().numpy().tolist())

    acc = accuracy_score(all_labels, all_preds)
    return sum(all_losses) / len(all_losses), acc


def eval_one_epoch(model, loader, criterion, device, label_to_idx):
    model.eval()
    all_losses = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y, lengths in tqdm(loader, desc="Val", leave=False):
            x = x.to(device)
            y_mapped = _map_labels(y, label_to_idx, device)

            logits = model(x, lengths)
            loss = criterion(logits, y_mapped)

            all_losses.append(loss.item())
            preds = logits.argmax(dim=1).detach().cpu().numpy().tolist()
            all_preds.extend(preds)
            all_labels.extend(y_mapped.detach().cpu().numpy().tolist())

    acc = accuracy_score(all_labels, all_preds)
    return sum(all_losses) / len(all_losses), acc


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    base_dataset = UCF101SkeletonDataset(
        pkl_path=args.pkl_path,
        split_name=args.split_name,
        selected_labels=args.selected_labels,
        max_frames=args.max_frames,
    )

    input_dim = 2 * base_dataset.num_joints

    if args.selected_labels is not None:
        unique_labels: List[int] = sorted(list(set(args.selected_labels)))
    else:
        unique_labels = sorted({int(ann["label"]) for ann in base_dataset.samples})

    label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}
    num_classes = len(unique_labels)

    train_ds, val_ds = train_val_split(base_dataset, val_ratio=args.val_ratio)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    if args.model_type == "baseline":
        model_kwargs = {
            "input_dim": input_dim,
            "num_classes": num_classes,
            "hidden_dim": args.baseline_hidden_dim,
        }
        model = TemporalMeanMLP(**model_kwargs)
        print(f"Modelo baseline → hidden_dim={args.baseline_hidden_dim}, max_frames={args.max_frames}")
    else:
        model_kwargs = {
            "input_dim": input_dim,
            "num_classes": num_classes,
            "hidden_dim": args.lstm_hidden_dim,
            "num_layers": args.lstm_layers,
            "bidirectional": args.bidirectional,
            "dropout": args.dropout,
        }
        model = LSTMSkeletonClassifier(**model_kwargs)
        print(
            "Modelo LSTM → "
            f"hidden_dim={args.lstm_hidden_dim}, layers={args.lstm_layers}, "
            f"bidir={args.bidirectional}, dropout={args.dropout}, max_frames={args.max_frames}"
        )

    device = args.device
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None
    if args.scheduler_step and args.scheduler_step > 0:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma
        )
    print(
        f"Optimizador → lr={args.lr}, weight_decay={args.weight_decay}, "
        f"grad_clip={args.grad_clip}, scheduler_step={args.scheduler_step}, "
        f"scheduler_gamma={args.scheduler_gamma}"
    )

    best_val_acc = 0.0
    best_path = os.path.join(args.output_dir, f"best_{args.model_type}.pt")

    for epoch in range(1, args.num_epochs + 1):
        print(f"\nEpoch {epoch}/{args.num_epochs}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, label_to_idx, grad_clip=args.grad_clip
        )

        val_loss, val_acc = eval_one_epoch(model, val_loader, criterion, device, label_to_idx)

        print(
            f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f} | "
            f"Val loss: {val_loss:.4f} | Val acc: {val_acc:.4f}"
        )

        if scheduler is not None:
            scheduler.step()

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
                    "model_kwargs": model_kwargs,
                },
                best_path,
            )
            print(f"Nuevo mejor modelo guardado en {best_path} (val_acc={best_val_acc:.4f})")

    print(f"\nMejor val_acc: {best_val_acc:.4f}")
    print(f"Modelo final guardado en: {best_path}")


if __name__ == "__main__":
    main()
