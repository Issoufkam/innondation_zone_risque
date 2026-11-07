"""Applique le U-Net entraîné sur une scène complète (pas seulement les
vignettes de test) pour produire une carte d'exposition : bâti détecté,
eau/zone inondée détectée, et bâti situé à moins de 50 m d'une zone d'eau
(indicateur d'exposition). Exporte un GeoTIFF et calcule les surfaces
(hectares) — aucune valeur n'est inventée, tout est calculé depuis les
pixels prédits.
"""

import argparse
import json

import numpy as np
import rasterio
import torch
from scipy.ndimage import binary_dilation

from config import CHECKPOINTS_DIR, CLASS_BUILDING, CLASS_WATER, PROCESSED, RAW, REPORTS_DIR, TILE_SIZE
from tiling import DERIVED_BANDS, FEATURE_BANDS
from unet import UNetLeger

IGNORE_LABEL = 255
DISTANCE_RISQUE_M = 50  # bâti considéré exposé si à moins de 50 m d'une zone d'eau détectée
RESOLUTION_M = 10
RAYON_PIXELS = max(1, round(DISTANCE_RISQUE_M / RESOLUTION_M))


def charger_pile(scene_id):
    arrays = []
    for band in FEATURE_BANDS:
        with rasterio.open(RAW / scene_id / f"{band}.tif") as src:
            arrays.append(src.read(1).astype("float32") / 10000.0)
            profile = src.profile.copy()
    for band in DERIVED_BANDS:
        with rasterio.open(PROCESSED / scene_id / f"{band}.tif") as src:
            arrays.append(src.read(1).astype("float32"))
    with rasterio.open(PROCESSED / scene_id / "masque_nuages.tif") as src:
        cloud_mask = src.read(1).astype(bool)
    return np.stack(arrays, axis=0), cloud_mask, profile


def inference_pleine_scene(modele, stack):
    _, height, width = stack.shape
    pred = np.zeros((height, width), dtype="uint8")
    with torch.no_grad():
        for y0 in range(0, height, TILE_SIZE):
            for x0 in range(0, width, TILE_SIZE):
                y1, x1 = min(y0 + TILE_SIZE, height), min(x0 + TILE_SIZE, width)
                tuile = stack[:, y0:y1, x0:x1]
                h, w = tuile.shape[1], tuile.shape[2]
                if h < TILE_SIZE or w < TILE_SIZE:  # complète le bord par du zéro-padding
                    pad = np.zeros((tuile.shape[0], TILE_SIZE, TILE_SIZE), dtype="float32")
                    pad[:, :h, :w] = tuile
                    tuile = pad
                entree = torch.from_numpy(tuile).unsqueeze(0)
                sortie = modele(entree).argmax(dim=1).squeeze(0).numpy()
                pred[y0:y1, x0:x1] = sortie[:h, :w]
    return pred


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default=None, help="Identifiant de scène (par défaut : la moins nuageuse)")
    args = parser.parse_args()

    index = json.loads((RAW / "index.json").read_text(encoding="utf-8"))
    scene = args.scene
    if scene is None:
        scene = min(index, key=lambda s: s["cloud_cover"])["scene_id"]
    print(f"Scène utilisée : {scene}")

    ckpt = torch.load(CHECKPOINTS_DIR / "unet.pt", map_location="cpu", weights_only=True)
    modele = UNetLeger(in_channels=ckpt["in_channels"], n_classes=3)
    modele.load_state_dict(ckpt["state_dict"])
    modele.eval()

    stack, cloud_mask, profile = charger_pile(scene)
    pred = inference_pleine_scene(modele, stack)
    pred[cloud_mask] = IGNORE_LABEL

    bati = pred == CLASS_BUILDING
    eau = pred == CLASS_WATER
    zone_risque = binary_dilation(eau, iterations=RAYON_PIXELS)
    bati_expose = bati & zone_risque

    pixel_ha = (RESOLUTION_M * RESOLUTION_M) / 10_000
    stats = {
        "scene_id": scene,
        "surface_bati_totale_ha": float(bati.sum() * pixel_ha),
        "surface_eau_inondation_ha": float(eau.sum() * pixel_ha),
        "surface_bati_expose_ha": float(bati_expose.sum() * pixel_ha),
        "part_bati_expose_pct": float(100 * bati_expose.sum() / max(bati.sum(), 1)),
        "distance_risque_m": DISTANCE_RISQUE_M,
    }
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    with open(REPORTS_DIR / "exposition.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # Carte à 4 valeurs : 0 fond, 1 bâti, 2 eau, 3 bâti exposé, 255 ignoré.
    carte = pred.copy()
    carte[bati_expose] = 3
    out_profile = profile.copy()
    out_profile.update(dtype="uint8", count=1, nodata=IGNORE_LABEL)
    out_path = REPORTS_DIR / f"carte_exposition_{scene}.tif"
    with rasterio.open(out_path, "w", **out_profile) as dst:
        dst.write(carte, 1)
    print(f"Carte d'exposition écrite dans {out_path}")


if __name__ == "__main__":
    main()
