"""Dataset PyTorch qui lit les vignettes .npz produites par tiling.py."""

import json

import numpy as np
import torch
from torch.utils.data import Dataset

from config import PROCESSED

IGNORE_LABEL = 255


class VignettesInondation(Dataset):
    def __init__(self, split: str):
        manifest = json.loads((PROCESSED / "tiles" / "manifest.json").read_text(encoding="utf-8"))
        self.fichiers = [PROCESSED / rel for rel in manifest[split]]
        if not self.fichiers:
            raise RuntimeError(f"Aucune vignette pour le split « {split} ». Lancez tiling.py.")

    def __len__(self):
        return len(self.fichiers)

    def __getitem__(self, idx):
        data = np.load(self.fichiers[idx])
        bands = torch.from_numpy(data["bands"])  # (C, H, W) float32
        label = torch.from_numpy(data["label"].astype("int64"))  # (H, W)
        return bands, label
