"""Recherche et télécharge les bandes Sentinel-2 L2A (Digital Earth Africa)
sur la zone d'étude, en ne lisant que la fenêtre qui recouvre l'AOI (évite
de télécharger des scènes entières de plusieurs centaines de Mo).

Les bandes vert/PIR sont nativement à 10 m, les bandes SWIR et la SCL à
20 m : ces dernières sont reprojetées (resample) sur la grille 10 m de la
bande verte pour obtenir une pile de bandes parfaitement alignée.

Source : catalogue STAC public https://explorer.digitalearth.africa/stac
(collection s2_l2a_c1), données hébergées sur un bucket S3 public en lecture
anonyme (deafrica-sentinel-2-l2a-c1, af-south-1). Aucun compte requis.
"""

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np
import rasterio
import requests
from rasterio.enums import Resampling
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import from_bounds

from config import AOI_BBOX, DEA_COLLECTION, DEA_STAC_URL, RAW, S2_BANDS

S3_HREF_RE = re.compile(r"^s3://([^/]+)/(.+)$")
BANDE_REFERENCE = "green"  # définit la grille 10 m de sortie
BANDES_CATEGORIELLES = {"scl"}


def s3_to_https(href: str) -> str:
    """Convertit un href s3://bucket/key en URL HTTPS publique (af-south-1)."""
    m = S3_HREF_RE.match(href)
    if not m:
        return href
    bucket, key = m.groups()
    return f"https://{bucket}.s3.af-south-1.amazonaws.com/{key}"


def search_scenes(start: str, end: str, max_cloud: float, limit: int):
    body = {
        "collections": [DEA_COLLECTION],
        "bbox": list(AOI_BBOX),
        "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z",
        "limit": 100,
    }
    resp = requests.post(DEA_STAC_URL, json=body, timeout=60)
    resp.raise_for_status()
    features = resp.json().get("features", [])

    def cloud(f):
        return f["properties"].get("eo:cloud_cover", 100)

    features = [f for f in features if cloud(f) <= max_cloud]
    features.sort(key=cloud)
    return features[:limit]


def lire_fenetre_reference(href: str, dest: Path):
    """Lit la bande de référence (10 m) sur l'AOI et fige la grille de sortie.

    L'AOI chevauche parfois deux tuiles MGRS voisines : la recherche STAC
    renvoie alors aussi des scènes qui ne couvrent qu'un petit bord de la
    zone. On les écarte en comparant la fenêtre lue à la taille attendue
    (calculée depuis l'AOI et la résolution, indépendamment de la tuile).
    """
    url = s3_to_https(href)
    with rasterio.open(f"/vsicurl/{url}") as src:
        aoi_in_src_crs = transform_bounds("EPSG:4326", src.crs, *AOI_BBOX)
        window = from_bounds(*aoi_in_src_crs, transform=src.transform).round_offsets().round_lengths()
        if window.width <= 0 or window.height <= 0:
            print("  (hors zone) scène ignorée")
            return None

        largeur_attendue = (aoi_in_src_crs[2] - aoi_in_src_crs[0]) / abs(src.transform.a)
        hauteur_attendue = (aoi_in_src_crs[3] - aoi_in_src_crs[1]) / abs(src.transform.e)

        # src.read() avec une fenêtre qui dépasse l'étendue réelle du raster
        # la découpe silencieusement (lecture plus petite que la fenêtre
        # demandée) : c'est ce qui révèle une tuile qui ne couvre qu'un bord
        # de l'AOI, pas la taille de la fenêtre elle-même.
        data = src.read(1, window=window)
        if data.shape[1] < 0.9 * largeur_attendue or data.shape[0] < 0.9 * hauteur_attendue:
            print(
                f"  (couverture partielle de l'AOI : {data.shape[1]}x{data.shape[0]} sur "
                f"{largeur_attendue:.0f}x{hauteur_attendue:.0f} attendu — tuile voisine, ignorée)"
            )
            return None
        transform = src.window_transform(window)
        profile = src.profile.copy()
        profile.update(height=data.shape[0], width=data.shape[1], transform=transform)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(dest, "w", **profile) as dst:
            dst.write(data, 1)
    print(f"  téléchargé (référence 10 m) : {dest.name} ({data.shape[1]}x{data.shape[0]})")
    return profile


def reprojeter_sur_reference(href: str, dest: Path, ref_profile: dict, categorique: bool):
    if dest.exists():
        print(f"  déjà présent : {dest.name}")
        return
    url = s3_to_https(href)
    with rasterio.open(f"/vsicurl/{url}") as src:
        destination = np.zeros((ref_profile["height"], ref_profile["width"]), dtype=src.dtypes[0])
        reproject(
            source=rasterio.band(src, 1),
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_profile["transform"],
            dst_crs=ref_profile["crs"],
            resampling=Resampling.nearest if categorique else Resampling.bilinear,
        )
        profile = ref_profile.copy()
        profile.update(dtype=destination.dtype)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(dest, "w", **profile) as dst:
            dst.write(destination, 1)
    print(f"  téléchargé (reprojeté sur grille 10 m) : {dest.name}")


def download_scene_bands(feat, scene_dir: Path) -> bool:
    """Retourne False si la scène ne couvre pas assez l'AOI (et doit être écartée)."""
    bandes_ordonnees = [BANDE_REFERENCE] + [b for b in S2_BANDS if b != BANDE_REFERENCE]
    ref_profile = None
    for band in bandes_ordonnees:
        asset = feat["assets"].get(band)
        if not asset:
            print(f"  bande absente : {band}")
            continue
        dest = scene_dir / f"{band}.tif"
        if band == BANDE_REFERENCE:
            if dest.exists():
                with rasterio.open(dest) as src:
                    ref_profile = src.profile.copy()
                print(f"  déjà présent (référence) : {dest.name}")
            else:
                ref_profile = lire_fenetre_reference(asset["href"], dest)
                if ref_profile is None:
                    return False
        else:
            reprojeter_sur_reference(asset["href"], dest, ref_profile, band in BANDES_CATEGORIELLES)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2023-06-01", help="Date de début (saison des pluies conseillée)")
    parser.add_argument("--end", default="2023-10-31", help="Date de fin")
    parser.add_argument("--max-cloud", type=float, default=30.0, help="Couverture nuageuse max (%)")
    parser.add_argument("--limit", type=int, default=4, help="Nombre de scènes à télécharger")
    args = parser.parse_args()

    print(f"Recherche Sentinel-2 L2A ({args.start} -> {args.end}), nébulosité <= {args.max_cloud}%...")
    scenes = search_scenes(args.start, args.end, args.max_cloud, args.limit)
    print(f"{len(scenes)} scène(s) retenue(s).")

    index = []
    for feat in scenes:
        scene_id = feat["id"]
        date = feat["properties"]["datetime"][:10]
        cloud = feat["properties"].get("eo:cloud_cover")
        print(f"- {scene_id} ({date}, nuages {cloud}%)")
        scene_dir = RAW / scene_id
        if download_scene_bands(feat, scene_dir):
            index.append({"scene_id": scene_id, "date": date, "cloud_cover": cloud, "dir": str(scene_dir)})
        elif scene_dir.exists():
            shutil.rmtree(scene_dir)

    # Fusionne avec les scènes déjà indexées précédemment (téléchargements cumulatifs).
    existing = []
    if (RAW / "index.json").exists():
        existing = json.loads((RAW / "index.json").read_text(encoding="utf-8"))
    vus = {s["scene_id"] for s in index}
    index = index + [s for s in existing if s["scene_id"] not in vus]

    with open(RAW / "index.json", "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    print(f"\nIndex écrit dans {RAW / 'index.json'}")


if __name__ == "__main__":
    main()
