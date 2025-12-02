import argparse
import random
from typing import Optional, Tuple

import numpy as np
import torch

from datasets import UCF101SkeletonDataset
from model import TemporalMeanMLP, LSTMSkeletonClassifier


def parse_args():
    parser = argparse.ArgumentParser(description="Demo de predicción en UCF101 Skeleton")
    parser.add_argument("--pkl_path", type=str, default="data/ucf101_2d.pkl")
    parser.add_argument("--split_name", type=str, default="test1",
                        help="Split del pkl a usar para demo (ej: 'test1', 'test2', 'test3').")
    parser.add_argument("--selected_labels", type=int, nargs="+", default=None)
    parser.add_argument("--max_frames", type=int, default=64)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_lstm.pt")
    parser.add_argument("--index", type=int, default=None, help="Índice específico dentro del split.")
    parser.add_argument("--frame_dir", type=str, default=None,
                        help="Nombre exacto de frame_dir para predecir esa muestra.")
    parser.add_argument("--keypoint_path", type=str, default=None,
                        help="Ruta a .npy/.npz con keypoints T x V x 2 para predicción directa.")
    parser.add_argument("--img_shape", type=int, nargs=2, default=None,
                        help="Altura y ancho si los keypoints vienen en pixeles.")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _extract_class_names(ds: UCF101SkeletonDataset):
    names = {}
    for ann in ds.samples:
        label = int(ann["label"])
        parts = ann["frame_dir"].split("_")
        names[label] = parts[1] if len(parts) > 1 else ann["frame_dir"]
    return names


def _find_index_by_frame_dir(ds: UCF101SkeletonDataset, frame_dir: str) -> Optional[int]:
    for idx, ann in enumerate(ds.samples):
        if ann["frame_dir"] == frame_dir:
            return idx
    return None


def _pad_or_crop(seq: np.ndarray, max_frames: int) -> Tuple[np.ndarray, int]:
    T, D = seq.shape
    if T == max_frames:
        return seq, T
    if T > max_frames:
        start = max(0, (T - max_frames) // 2)
        end = start + max_frames
        return seq[start:end], max_frames
    pad_len = max_frames - T
    pad = np.zeros((pad_len, D), dtype=seq.dtype)
    return np.concatenate([seq, pad], axis=0), T


def _prepare_custom_sequence(
    path: str,
    img_shape: Optional[Tuple[int, int]],
    max_frames: int,
    num_joints: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    arr = np.load(path, allow_pickle=True)
    if isinstance(arr, np.lib.npyio.NpzFile):
        key = "keypoints" if "keypoints" in arr else arr.files[0]
        arr = arr[key]

    if arr.ndim != 3 or arr.shape[2] != 2:
        raise ValueError("Se esperaba arreglo T x V x 2 para keypoints.")
    if arr.shape[1] != num_joints:
        raise ValueError(f"num_joints esperado {num_joints}, recibido {arr.shape[1]}")

    kp = arr.astype(np.float32)
    if img_shape is not None:
        h, w = img_shape
        kp[..., 0] /= float(w)
        kp[..., 1] /= float(h)

    seq = kp.reshape(kp.shape[0], num_joints * 2)
    seq_fixed, seq_len = _pad_or_crop(seq, max_frames)
    seq_len = min(seq_len, max_frames)

    return torch.from_numpy(seq_fixed).float(), torch.tensor(seq_len, dtype=torch.long)


def main():
    args = parse_args()
    device = args.device

    ckpt = torch.load(args.checkpoint, map_location=device)
    model_type = ckpt["model_type"]
    input_dim = ckpt["input_dim"]
    num_classes = ckpt["num_classes"]
    label_to_idx = ckpt["label_to_idx"]
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    max_frames = ckpt.get("max_frames", args.max_frames)
    num_joints = input_dim // 2

    model_kwargs = ckpt.get("model_kwargs", {})
    if model_type == "baseline":
        default_kwargs = {"input_dim": input_dim, "num_classes": num_classes}
        default_kwargs.update(model_kwargs)
        model = TemporalMeanMLP(**default_kwargs)
    else:
        default_kwargs = {
            "input_dim": input_dim,
            "num_classes": num_classes,
            "hidden_dim": model_kwargs.get("hidden_dim", 256),
            "num_layers": model_kwargs.get("num_layers", 2),
            "bidirectional": model_kwargs.get("bidirectional", False),
            "dropout": model_kwargs.get("dropout", 0.3),
        }
        model = LSTMSkeletonClassifier(**default_kwargs)

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    ds = UCF101SkeletonDataset(
        pkl_path=args.pkl_path,
        split_name=args.split_name,
        selected_labels=list(label_to_idx.keys()),
        max_frames=max_frames,
    )
    label_names = _extract_class_names(ds)

    x_tensor: torch.Tensor
    seq_len: torch.Tensor
    y_original: Optional[int] = None

    if args.keypoint_path is not None:
        x_tensor, seq_len = _prepare_custom_sequence(
            args.keypoint_path,
            tuple(args.img_shape) if args.img_shape is not None else None,
            max_frames=max_frames,
            num_joints=num_joints,
        )
        source = f"archivo externo: {args.keypoint_path}"
    else:
        idx = args.index
        if args.frame_dir:
            idx = _find_index_by_frame_dir(ds, args.frame_dir)
            if idx is None:
                raise ValueError(f"No se encontró frame_dir '{args.frame_dir}' en el split {args.split_name}.")
        if idx is None:
            idx = random.randint(0, len(ds) - 1)

        x_tensor, y_original_tensor, seq_len = ds[idx]
        y_original = int(y_original_tensor.item())
        source = f"sample idx {idx} | frame_dir: {ds.samples[idx]['frame_dir']}"

    x = x_tensor.unsqueeze(0).to(device)
    lengths = seq_len.unsqueeze(0)

    with torch.no_grad():
        logits = model(x, lengths)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = int(torch.argmax(probs).item())

    pred_label_original = idx_to_label[pred_idx]
    pred_label_name = label_names.get(pred_label_original, "desconocido")

    print(f"Predicción sobre {source}")
    if y_original is not None:
        true_name = label_names.get(y_original, "desconocido")
        print(f"Label verdadero (UCF101 id): {y_original} | clase: {true_name}")
    print(f"Label predicho (UCF101 id): {pred_label_original} | clase: {pred_label_name}")

    topk = min(args.topk, num_classes)
    top_scores, top_indices = torch.topk(probs, k=topk)
    print("\nTop-k probabilidades:")
    for score, idx_pred in zip(top_scores.tolist(), top_indices.tolist()):
        lbl = idx_to_label[idx_pred]
        name = label_names.get(lbl, "desconocido")
        print(f"{score:.3f} -> {lbl} ({name})")


if __name__ == "__main__":
    main()
