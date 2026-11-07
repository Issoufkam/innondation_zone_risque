"""Calcule les indices spectraux MNDWI (eau) et NDBI (bâti) comme
heuristiques de référence (baselines), et un masque de nuages à partir de
la bande SCL (Scene Classification Layer) de Sentinel-2 L2A.

MNDWI (Xu, 2006) : (vert - SWIR) / (vert + SWIR) — plus robuste que le NDWI
classique en milieu urbain/turbide.
NDBI (Zha et al., 2003) : (SWIR - PIR) / (SWIR + PIR).

Valeurs SCL à exclure (nuages, ombres, nodata) : voir SCL_INVALIDES.
"""

import json

import numpy as np
import rasterio

from config import PROCESSED, RAW

# Classes SCL Sentinel-2 L2A à exclure : 0 nodata, 1 saturé/défectueux,
# 3 ombre de nuage, 8/9 nuage moyen/fort, 10 cirrus fin, 11 neige/glace.
SCL_INVALIDES = {0, 1, 3, 8, 9, 10, 11}


def read_band(path):
    with rasterio.open(path) as src:
        return src.read(1).astype("float32"), src.profile


def main():
    index_path = RAW / "index.json"
    if not index_path.exists():
        raise SystemExit("Lancez d'abord download_sentinel2.py")
    scenes = json.loads(index_path.read_text(encoding="utf-8"))

    for scene in scenes:
        scene_dir = RAW.parent / "raw" / scene["scene_id"]
        out_dir = PROCESSED / scene["scene_id"]
        out_dir.mkdir(parents=True, exist_ok=True)

        green, profile = read_band(scene_dir / "green.tif")
        nir, _ = read_band(scene_dir / "nir.tif")
        swir16, _ = read_band(scene_dir / "swir16.tif")
        scl, _ = read_band(scene_dir / "scl.tif")

        # Réflectance Sentinel-2 L2A stockée en entier (x10000).
        green, nir, swir16 = green / 10000.0, nir / 10000.0, swir16 / 10000.0

        eps = 1e-6
        mndwi = (green - swir16) / (green + swir16 + eps)
        ndbi = (swir16 - nir) / (swir16 + nir + eps)
        cloud_mask = np.isin(scl.astype("int32"), list(SCL_INVALIDES))

        out_profile = profile.copy()
        out_profile.update(dtype="float32", count=1, nodata=None)
        for name, arr in [("mndwi", mndwi), ("ndbi", ndbi)]:
            with rasterio.open(out_dir / f"{name}.tif", "w", **out_profile) as dst:
                dst.write(arr.astype("float32"), 1)

        mask_profile = profile.copy()
        mask_profile.update(dtype="uint8", count=1, nodata=255)
        with rasterio.open(out_dir / "masque_nuages.tif", "w", **mask_profile) as dst:
            dst.write(cloud_mask.astype("uint8"), 1)

        nuages_pct = 100 * cloud_mask.mean()
        print(f"{scene['scene_id']} : MNDWI/NDBI calculés, {nuages_pct:.1f}% de pixels nuageux/invalides")


if __name__ == "__main__":
    main()
