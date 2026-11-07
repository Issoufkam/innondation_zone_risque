"""Évalue le U-Net sur le jeu de test (IoU, F1-score par classe) et
exporte les masques prédits en GeoTIFF.
"""

import json

import numpy as np
import rasterio
import torch
from affine import Affine
from torch.utils.data import DataLoader

from config import CHECKPOINTS_DIR, CLASS_NAMES, MASKS_DIR, PROCESSED, REPORTS_DIR
from dataset import VignettesInondation
from unet import UNetLeger

IGNORE_LABEL = 255
N_CLASSES = 3


def iou_f1_par_classe(matrice_confusion):
    resultats = {}
    for c in range(N_CLASSES):
        tp = matrice_confusion[c, c]
        fp = matrice_confusion[:, c].sum() - tp
        fn = matrice_confusion[c, :].sum() - tp
        iou = tp / max(tp + fp + fn, 1)
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        resultats[CLASS_NAMES[c]] = {"iou": float(iou), "f1": float(f1)}
    return resultats


def main():
    ckpt = torch.load(CHECKPOINTS_DIR / "unet.pt", map_location="cpu", weights_only=True)
    modele = UNetLeger(in_channels=ckpt["in_channels"], n_classes=N_CLASSES)
    modele.load_state_dict(ckpt["state_dict"])
    modele.eval()

    test_ds = VignettesInondation("test")
    loader = DataLoader(test_ds, batch_size=8, shuffle=False)

    matrice = np.zeros((N_CLASSES, N_CLASSES), dtype="int64")
    with torch.no_grad():
        for bands, label in loader:
            sortie = modele(bands)
            pred = sortie.argmax(dim=1).numpy()
            label_np = label.numpy()
            valide = label_np != IGNORE_LABEL
            for p, l in zip(pred[valide], label_np[valide]):
                matrice[l, p] += 1

    resultats = iou_f1_par_classe(matrice)
    print(json.dumps(resultats, indent=2, ensure_ascii=False))
    with open(REPORTS_DIR / "evaluation.json", "w", encoding="utf-8") as f:
        json.dump(resultats, f, indent=2, ensure_ascii=False)

    print("\nExport des masques prédits (jeu de test) en GeoTIFF...")
    manifest = json.loads((PROCESSED / "tiles" / "manifest.json").read_text(encoding="utf-8"))
    with torch.no_grad():
        for rel_path in manifest["test"]:
            tile_path = PROCESSED / rel_path
            data = np.load(tile_path)
            bands = torch.from_numpy(data["bands"]).unsqueeze(0)
            pred = modele(bands).argmax(dim=1).squeeze(0).numpy().astype("uint8")

            transform = Affine(*data["transform"])
            crs = str(data["crs"])
            out_path = MASKS_DIR / (tile_path.stem + "_predit.tif")
            profile = {
                "driver": "GTiff",
                "height": pred.shape[0],
                "width": pred.shape[1],
                "count": 1,
                "dtype": "uint8",
                "crs": crs,
                "transform": transform,
            }
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(pred, 1)
    print(f"Masques écrits dans {MASKS_DIR}")


if __name__ == "__main__":
    main()
