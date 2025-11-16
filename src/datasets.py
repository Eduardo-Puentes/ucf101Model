# src/datasets.py
import pickle
from typing import List, Optional, Dict, Any

import numpy as np
import torch
from torch.utils.data import Dataset, random_split


class UCF101SkeletonDataset(Dataset):
    """
    Dataset para UCF101 con esqueletos 2D en formato MMAction2.

    Espera un .pkl con campos:
      - 'split': dict con splits → lista de frame_dir
      - 'annotations': lista de dicts con:
          'frame_dir', 'total_frames', 'img_shape', 'label',
          'keypoint' (M x T x V x C), 'keypoint_score' (M x T x V)
    """

    def __init__(
        self,
        pkl_path: str,
        split_name: str,
        selected_labels: Optional[List[int]] = None,
        max_frames: int = 64,
    ):
        super().__init__()

        with open(pkl_path, "rb") as f:
            data = pickle.load(f, encoding="latin1")  # latin1 por compatibilidad

        self.splits: Dict[str, List[str]] = data["split"]
        self.annotations: List[Dict[str, Any]] = data["annotations"]

        if split_name not in self.splits:
            raise ValueError(
                f"El split '{split_name}' no existe. Splits disponibles: {list(self.splits.keys())}"
            )

        allowed_frame_dirs = set(self.splits[split_name])

        self.max_frames = max_frames
        self.selected_labels = set(selected_labels) if selected_labels is not None else None

        self.samples = []
        for ann in self.annotations:
            fd = ann["frame_dir"]
            if fd not in allowed_frame_dirs:
                continue
            label = int(ann["label"])
            if self.selected_labels is not None and label not in self.selected_labels:
                continue
            self.samples.append(ann)

        if len(self.samples) == 0:
            raise RuntimeError(
                "No se encontraron muestras para ese split / selected_labels. "
                "Revisa los IDs de clase y los nombres de split."
            )

        # Determinamos V (número de joints) a partir de la primera muestra
        example_kp = self.samples[0]["keypoint"]  # shape: M x T x V x C
        _, _, V, C = example_kp.shape
        assert C == 2, "Se esperaban keypoints 2D (C=2)."
        self.num_joints = V

    def __len__(self) -> int:
        return len(self.samples)

    def _select_main_person(self, keypoint: np.ndarray, keypoint_score: np.ndarray):
        """
        keypoint: M x T x V x C
        keypoint_score: M x T x V

        Devuelve el skeleton de la persona con score promedio más alto: T x V x C
        """
        M = keypoint.shape[0]
        if M == 1:
            return keypoint[0]  # T x V x C

        avg_scores = keypoint_score.mean(axis=(1, 2))  # M
        main_idx = int(np.argmax(avg_scores))
        return keypoint[main_idx]  # T x V x C

    def _normalize_coords(self, kp: np.ndarray, img_shape):
        """
        kp: T x V x C, donde C=2 (x, y)
        img_shape: (h, w)
        Normaliza a [0, 1] por ancho/alto.
        """
        h, w = img_shape
        kp_norm = kp.copy().astype(np.float32)
        kp_norm[..., 0] /= float(w)
        kp_norm[..., 1] /= float(h)
        return kp_norm

    def _pad_or_crop(self, seq: np.ndarray) -> np.ndarray:
        """
        seq: T x (2V)
        Devuelve: max_frames x (2V)
        """
        T, D = seq.shape
        if T == self.max_frames:
            return seq
        if T > self.max_frames:
            # recortamos centrado aproximadamente
            start = max(0, (T - self.max_frames) // 2)
            end = start + self.max_frames
            return seq[start:end]
        # padding con ceros al final
        pad_len = self.max_frames - T
        pad = np.zeros((pad_len, D), dtype=seq.dtype)
        return np.concatenate([seq, pad], axis=0)

    def __getitem__(self, idx: int):
        ann = self.samples[idx]
        label = int(ann["label"])
        img_shape = ann["img_shape"]  # (h, w)

        keypoint = ann["keypoint"]          # M x T x V x C
        keypoint_score = ann["keypoint_score"]  # M x T x V

        # Seleccionamos a la persona principal
        kp_main = self._select_main_person(keypoint, keypoint_score)  # T x V x C

        # Normalizamos coordenadas
        kp_main = self._normalize_coords(kp_main, img_shape)  # T x V x C

        # Reordenamos a (T x 2V)
        T, V, C = kp_main.shape
        seq = kp_main.reshape(T, V * C)  # T x (2V)

        # Padding / crop a longitud fija
        seq_fixed = self._pad_or_crop(seq)  # max_frames x (2V)

        # Convertimos a tensores
        seq_tensor = torch.from_numpy(seq_fixed).float()
        label_tensor = torch.tensor(label, dtype=torch.long)

        return seq_tensor, label_tensor


def train_val_split(
    dataset: UCF101SkeletonDataset,
    val_ratio: float = 0.2,
    seed: int = 42,
):
    """
    Split aleatorio train/val a partir de un dataset de un solo split.
    Si tus splits ya vienen separados en el pkl, puedes no usar esto.
    """
    num_total = len(dataset)
    num_val = int(num_total * val_ratio)
    num_train = num_total - num_val
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [num_train, num_val], generator=generator)
    return train_ds, val_ds