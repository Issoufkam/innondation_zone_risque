"""Découpe chaque scène en vignettes (256x256) et sépare train/val/test
par blocs spatiaux contigus (pas de mélange aléatoire) pour éviter toute
fuite d'information entre les jeux : les 70% de colonnes à gauche vont à
l'entraînement, les 15% suivantes à la validation, les 15% de droite au
test.

Les bandes empilées (vert, PIR, SWIR16, SWIR22, MNDWI, NDBI) et
l'étiquette correspondante sont enregistrées ensemble dans un fichier
.npz par vignette, avec la transformation géographique (pour recomposer
une carte à l'évaluation).
"""

import json

import numpy as np
import rasterio

from config import PROCESSED, RAW, TILE_SIZE

FEATURE_BANDS = ["green", "nir", "swir16", "swir22"]  # bandes brutes (raw/)
DERIVED_BANDS = ["mndwi", "ndbi"]  # bandes dérivées (processed/)
IGNORE_LABEL = 255
MAX_IGNORE_RATIO = 0.5  # vignette écartée si plus de 50% nuages/invalide


def load_stack(scene_id):
    arrays = []
    ref_profile = None
    for band in FEATURE_BANDS:
        with rasterio.open(RAW / scene_id / f"{band}.tif") as src:
            arrays.append(src.read(1).astype("float32") / 10000.0)
            if ref_profile is None:
                ref_profile = src.profile
    for band in DERIVED_BANDS:
        with rasterio.open(PROCESSED / scene_id / f"{band}.tif") as src:
            arrays.append(src.read(1).astype("float32"))
    with rasterio.open(PROCESSED / scene_id / "label.tif") as src:
        label = src.read(1)
        transform = src.transform
        crs = src.crs
    stack = np.stack(arrays, axis=0)  # (C, H, W)
    return stack, label, transform, crs


def main():
    scenes = json.loads((RAW / "index.json").read_text(encoding="utf-8"))
    out_root = PROCESSED / "tiles"
    manifest = {"train": [], "val": [], "test": []}

    for scene in scenes:
        scene_id = scene["scene_id"]
        stack, label, transform, crs = load_stack(scene_id)
        _, height, width = stack.shape
        n_cols = width // TILE_SIZE
        n_rows = height // TILE_SIZE
        if n_cols == 0 or n_rows == 0:
            print(f"{scene_id} : trop petite pour une vignette ({width}x{height}), ignorée")
            continue

        train_limit = int(n_cols * 0.70)
        val_limit = int(n_cols * 0.85)

        n_garde, n_ecarte = 0, 0
        for row in range(n_rows):
            for col in range(n_cols):
                y0, x0 = row * TILE_SIZE, col * TILE_SIZE
                tile_bands = stack[:, y0 : y0 + TILE_SIZE, x0 : x0 + TILE_SIZE]
                tile_label = label[y0 : y0 + TILE_SIZE, x0 : x0 + TILE_SIZE]

                ignore_ratio = (tile_label == IGNORE_LABEL).mean()
                if ignore_ratio > MAX_IGNORE_RATIO:
                    n_ecarte += 1
                    continue

                split = "train" if col < train_limit else ("val" if col < val_limit else "test")
                split_dir = out_root / split
                split_dir.mkdir(parents=True, exist_ok=True)
                tile_transform = transform * transform.translation(x0, y0)
                name = f"{scene_id}_r{row}_c{col}.npz"
                np.savez_compressed(
                    split_dir / name,
                    bands=tile_bands.astype("float32"),
                    label=tile_label.astype("uint8"),
                    transform=np.array(tile_transform)[:6],
                    crs=str(crs),
                )
                manifest[split].append(str((split_dir / name).relative_to(PROCESSED)))
                n_garde += 1

        print(f"{scene_id} : {n_garde} vignette(s) gardée(s), {n_ecarte} écartée(s) (nuages)")

    with open(out_root / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    for split, items in manifest.items():
        print(f"{split} : {len(items)} vignette(s)")


if __name__ == "__main__":
    main()
