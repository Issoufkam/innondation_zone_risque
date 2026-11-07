"""Construit les masques d'entraînement (3 classes : fond, bâti, eau) pour
chaque scène Sentinel-2.

- Classe « bâti » : rasterisation des empreintes OpenStreetMap
  (donnée vérifiée, indépendante de l'imagerie).
- Classe « eau / zone potentiellement inondée » : seuillage du MNDWI
  (MNDWI > 0), une heuristique spectrale classique — PAS une carte
  d'inondation validée sur le terrain. Le U-Net apprend ensuite à
  reproduire/généraliser ce signal à partir des bandes brutes ; toute
  publication doit mentionner cette limite méthodologique.
- Les pixels nuageux/invalides (masque_nuages) sont marqués en classe
  « ignorer » (255) et exclus de la fonction de perte à l'entraînement.
"""

import json

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize

from config import BUILDINGS_DIR, CLASS_BUILDING, CLASS_WATER, PROCESSED, RAW

MNDWI_SEUIL = 0.0
IGNORE_LABEL = 255

_CACHE_BATIMENTS = {}


def load_buildings_geometries(dst_crs):
    """Reprojection vectorisée (geopandas/pyproj) — bien plus rapide qu'une
    boucle Python sur les ~300k polygones OSM."""
    if dst_crs in _CACHE_BATIMENTS:
        return _CACHE_BATIMENTS[dst_crs]
    path = BUILDINGS_DIR / "batiments_abidjan.geojson"
    if not path.exists():
        raise SystemExit("Lancez d'abord download_buildings.py")
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    gdf = gdf.to_crs(dst_crs)
    geoms = list(gdf.geometry)
    _CACHE_BATIMENTS[dst_crs] = geoms
    return geoms


def main():
    index_path = RAW / "index.json"
    scenes = json.loads(index_path.read_text(encoding="utf-8"))

    for scene in scenes:
        scene_id = scene["scene_id"]
        proc_dir = PROCESSED / scene_id

        with rasterio.open(proc_dir / "mndwi.tif") as src:
            mndwi = src.read(1)
            profile = src.profile
            transform = src.transform
            shape = src.shape
            crs = src.crs

        with rasterio.open(proc_dir / "masque_nuages.tif") as src:
            cloud_mask = src.read(1).astype(bool)

        buildings = load_buildings_geometries(crs)
        building_mask = rasterize(
            [(geom, 1) for geom in buildings],
            out_shape=shape,
            transform=transform,
            fill=0,
            dtype="uint8",
        ).astype(bool)

        label = np.zeros(shape, dtype="uint8")
        label[mndwi > MNDWI_SEUIL] = CLASS_WATER
        label[building_mask] = CLASS_BUILDING  # le bâti prime sur l'eau en cas de conflit
        label[cloud_mask] = IGNORE_LABEL

        out_profile = profile.copy()
        out_profile.update(dtype="uint8", count=1, nodata=IGNORE_LABEL)
        with rasterio.open(proc_dir / "label.tif", "w", **out_profile) as dst:
            dst.write(label, 1)

        valides = label != IGNORE_LABEL
        total = valides.sum()
        pct_bati = 100 * (label == CLASS_BUILDING).sum() / max(total, 1)
        pct_eau = 100 * (label == CLASS_WATER).sum() / max(total, 1)
        print(f"{scene_id} : bâti {pct_bati:.2f}% | eau/inondation {pct_eau:.2f}% (pixels valides)")


if __name__ == "__main__":
    main()
