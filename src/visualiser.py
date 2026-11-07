"""Produit un aperçu visuel (PNG) : image satellite, étiquette de
référence et masque prédit, pour quelques vignettes du jeu de test."""

import json

import matplotlib.pyplot as plt
import numpy as np

from config import CLASS_NAMES, MASKS_DIR, PROCESSED, REPORTS_DIR

COULEURS = {0: (0.85, 0.85, 0.85), 1: (0.70, 0.15, 0.15), 2: (0.15, 0.35, 0.80), 255: (1, 1, 1)}


def colorier(label):
    out = np.zeros((*label.shape, 3))
    for cls, couleur in COULEURS.items():
        out[label == cls] = couleur
    return out


def main(n=4):
    manifest = json.loads((PROCESSED / "tiles" / "manifest.json").read_text(encoding="utf-8"))
    echantillon = manifest["test"][:n]

    fig, axes = plt.subplots(len(echantillon), 3, figsize=(9, 3 * len(echantillon)))
    for i, rel_path in enumerate(echantillon):
        data = np.load(PROCESSED / rel_path)
        bands = data["bands"]  # green, nir, swir16, swir22, mndwi, ndbi
        verite = data["label"]

        pred_path = MASKS_DIR / (rel_path.split("/")[-1].replace(".npz", "_predit.tif"))
        if pred_path.exists():
            import rasterio

            with rasterio.open(pred_path) as src:
                pred = src.read(1)
        else:
            pred = np.zeros_like(verite)

        composite = np.stack([bands[2], bands[1], bands[0]], axis=-1)  # SWIR/PIR/vert en pseudo-couleur
        composite = np.clip(composite * 2.5, 0, 1)

        axes[i, 0].imshow(composite)
        axes[i, 0].set_title("Sentinel-2 (pseudo-couleur)" if i == 0 else "")
        axes[i, 1].imshow(colorier(verite))
        axes[i, 1].set_title("Étiquette (OSM + MNDWI)" if i == 0 else "")
        axes[i, 2].imshow(colorier(pred))
        axes[i, 2].set_title("Prédiction U-Net" if i == 0 else "")
        for ax in axes[i]:
            ax.axis("off")

    fig.tight_layout()
    out_path = REPORTS_DIR / "apercu_test.png"
    fig.savefig(out_path, dpi=120)
    print(f"Aperçu écrit dans {out_path}")


if __name__ == "__main__":
    main()
