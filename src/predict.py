# src/predict.py
import argparse
import random

import torch

from datasets import UCF101SkeletonDataset
from model import TemporalMeanMLP, LSTMSkeletonClassifier


def parse_args():
    parser = argparse.ArgumentParser(description="Demo de predicción en UCF101 Skeleton")
    parser.add_argument("--pkl_path", type=str, default="data/ucf101_2d.pkl")
    parser.add_argument("--split_name", type=str, default="test",
                        help="Split del pkl a usar para demo (ej: 'test').")
    parser.add_argument("--selected_labels", type=int, nargs="+", default=None)
    parser.add_argument("--max_frames", type=int, default=64)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_lstm.pt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device

    # Cargamos checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    model_type = ckpt["model_type"]
    input_dim = ckpt["input_dim"]
    num_classes = ckpt["num_classes"]
    label_to_idx = ckpt["label_to_idx"]
    idx_to_label = {v: k for k, v in label_to_idx.items()}

    # Modelo
    if model_type == "baseline":
        model = TemporalMeanMLP(input_dim=input_dim, num_classes=num_classes)
    else:
        model = LSTMSkeletonClassifier(input_dim=input_dim, num_classes=num_classes)

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # Dataset
    ds = UCF101SkeletonDataset(
        pkl_path=args.pkl_path,
        split_name=args.split_name,
        selected_labels=list(label_to_idx.keys()),
        max_frames=args.max_frames,
    )

    # Elegimos muestra aleatoria
    idx = random.randint(0, len(ds) - 1)
    x, y_original = ds[idx]  # y_original = label original
    x = x.unsqueeze(0).to(device)  # (1, T, D)

    with torch.no_grad():
        logits = model(x)
        pred_idx = int(logits.argmax(dim=1).item())

    pred_label_original = idx_to_label[pred_idx]

    print(f"Índice de muestra: {idx}")
    print(f"Label verdadero (original UCF101): {y_original.item()}")
    print(f"Label predicho (original UCF101): {pred_label_original}")


if __name__ == "__main__":
    main()