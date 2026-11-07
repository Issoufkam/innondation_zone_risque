"""Télécharge les bandes Sentinel-2 L2A depuis la source officielle
Copernicus Data Space Ecosystem (CDSE), via son stockage compatible S3
("eodata"). Alternative à download_sentinel2.py (qui utilise Digital
Earth Africa, sans compte) : celle-ci utilise la source citée dans le
projet d'origine, mais nécessite un compte Copernicus et des clés S3.

Clés S3 : à générer sur https://eodata-s3keysmanager.dataspace.copernicus.eu/
puis à mettre dans .env (voir .env.example) — jamais commitées.

Même logique de fenêtrage et de reprojection que download_sentinel2.py :
seule la fenêtre qui recouvre l'AOI est lue (pas la scène entière), et les
bandes 20 m (SWIR, SCL) sont reprojetées sur la grille 10 m de la bande
verte.
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import boto3
import numpy as np
import rasterio
import requests
from rasterio.enums import Resampling
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import from_bounds

from config import AOI_BBOX, RAW, ROOT

CDSE_ODATA_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
CDSE_S3_ENDPOINT = "eodata.dataspace.copernicus.eu"
CDSE_BUCKET = "eodata"

# Tuile MGRS qui recouvre entièrement l'AOI (vérifié via Digital Earth
# Africa : T30NVM donne une fenêtre pleine 3329x2516, les tuiles voisines
# ne couvrent qu'un bord). Fixer la tuile évite de perdre du temps sur des
# scènes à couverture partielle.
TUILE_MGRS = "30NVM"

# Bandes CDSE (noms officiels) -> noms logiques utilisés par le reste du pipeline
# (identiques à ceux de download_sentinel2.py / config.S2_BANDS).
BANDES_CDSE = {"green": "B03", "nir": "B08", "swir16": "B11", "swir22": "B12", "scl": "SCL"}
BANDE_REFERENCE = "green"  # B03, 10 m : définit la grille de sortie
BANDES_CATEGORIELLES = {"scl"}


def verifier_identifiants():
    cle = os.environ.get("CDSE_S3_ACCESS_KEY")
    secret = os.environ.get("CDSE_S3_SECRET_KEY")
    if not cle or not secret:
        raise SystemExit(
            "Clés S3 Copernicus manquantes. Copier .env.example en .env et renseigner "
            "CDSE_S3_ACCESS_KEY / CDSE_S3_SECRET_KEY (voir le fichier pour l'URL)."
        )
    os.environ.setdefault("AWS_ACCESS_KEY_ID", cle)
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", secret)
    os.environ.setdefault("AWS_S3_ENDPOINT", CDSE_S3_ENDPOINT)
    os.environ.setdefault("AWS_VIRTUAL_HOSTING", "FALSE")
    os.environ.setdefault("AWS_HTTPS", "YES")
    return cle, secret


def client_s3(cle, secret):
    return boto3.session.Session().client(
        "s3",
        aws_access_key_id=cle,
        aws_secret_access_key=secret,
        endpoint_url=f"https://{CDSE_S3_ENDPOINT}",
        region_name="default",
    )


def search_scenes(start: str, end: str, max_cloud: float, limit: int):
    min_lon, min_lat, max_lon, max_lat = AOI_BBOX
    polygon = (
        f"POLYGON(({min_lon} {min_lat},{max_lon} {min_lat},{max_lon} {max_lat},"
        f"{min_lon} {max_lat},{min_lon} {min_lat}))"
    )
    filtre = (
        "Collection/Name eq 'SENTINEL-2' and contains(Name,'MSIL2A') and "
        f"OData.CSC.Intersects(area=geography'SRID=4326;{polygon}') and "
        f"ContentDate/Start gt {start}T00:00:00.000Z and ContentDate/Start lt {end}T23:59:59.000Z and "
        f"Attributes/OData.CSC.StringAttribute/any(a:a/Name eq 'tileId' and a/Value eq '{TUILE_MGRS}')"
    )
    params = {"$filter": filtre, "$top": 100, "$expand": "Attributes"}
    resp = requests.get(CDSE_ODATA_URL, params=params, timeout=60)
    resp.raise_for_status()
    produits = resp.json().get("value", [])

    def cloud_cover(p):
        for a in p.get("Attributes", []):
            if a.get("Name") == "cloudCover":
                return a.get("Value", 100)
        return 100

    produits = [p for p in produits if cloud_cover(p) <= max_cloud]
    produits.sort(key=cloud_cover)
    resultats = []
    for p in produits[:limit]:
        resultats.append(
            {
                "scene_id": p["Name"].replace(".SAFE", ""),
                "date": p["ContentDate"]["Start"][:10],
                "cloud_cover": cloud_cover(p),
                "s3_path": p["S3Path"].lstrip("/"),  # ex: eodata/Sentinel-2/MSI/L2A_N0500/...
            }
        )
    return resultats


# Chaque bande n'est cherchée que dans son dossier de résolution native :
# R10m et R20m contiennent toutes les deux une copie de B03/B08 (la version
# R20m est une copie resamplée à 20 m, pas la bande native — la confondre
# avec la version 10 m fausse toute la grille de référence).
DOSSIER_PAR_BANDE = {"B03": "R10m/", "B08": "R10m/", "B11": "R20m/", "B12": "R20m/", "SCL": "R20m/"}


def trouver_fichiers_bandes(s3, s3_path):
    """Liste IMG_DATA/R10m et R20m du granule et retourne {code_bande: clé_s3},
    chaque bande prise uniquement dans son dossier de résolution native."""
    prefix_granule = s3_path.split("eodata/", 1)[-1] + "/GRANULE/"
    resp = s3.list_objects_v2(Bucket=CDSE_BUCKET, Prefix=prefix_granule, Delimiter="/")
    granules = [cp["Prefix"] for cp in resp.get("CommonPrefixes", [])]
    if not granules:
        raise RuntimeError("Aucun granule trouvé dans le produit.")
    granule = granules[0]

    listages = {}
    fichiers = {}
    for code, dossier in DOSSIER_PAR_BANDE.items():
        if dossier not in listages:
            resp = s3.list_objects_v2(Bucket=CDSE_BUCKET, Prefix=granule + "IMG_DATA/" + dossier)
            listages[dossier] = [obj["Key"] for obj in resp.get("Contents", [])]
        suffixe = f"_{code}_{dossier.rstrip('/').lstrip('R')}.jp2"  # ex: _B03_10m.jp2
        for cle in listages[dossier]:
            if cle.endswith(suffixe):
                fichiers[code] = cle
                break
    return fichiers


def lire_fenetre_reference(vsi_path: str, dest: Path):
    with rasterio.open(vsi_path) as src:
        aoi_in_src_crs = transform_bounds("EPSG:4326", src.crs, *AOI_BBOX)
        window = from_bounds(*aoi_in_src_crs, transform=src.transform).round_offsets().round_lengths()
        if window.width <= 0 or window.height <= 0:
            print("  (hors zone) scène ignorée")
            return None

        largeur_attendue = (aoi_in_src_crs[2] - aoi_in_src_crs[0]) / abs(src.transform.a)
        hauteur_attendue = (aoi_in_src_crs[3] - aoi_in_src_crs[1]) / abs(src.transform.e)

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


def reprojeter_sur_reference(vsi_path: str, dest: Path, ref_profile: dict, categorique: bool):
    if dest.exists():
        print(f"  déjà présent : {dest.name}")
        return
    with rasterio.open(vsi_path) as src:
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


def download_scene_bands(s3, scene, scene_dir: Path) -> bool:
    fichiers = trouver_fichiers_bandes(s3, scene["s3_path"])
    bandes_ordonnees = [BANDE_REFERENCE] + [b for b in BANDES_CDSE if b != BANDE_REFERENCE]
    ref_profile = None
    for band in bandes_ordonnees:
        code = BANDES_CDSE[band]
        cle_s3 = fichiers.get(code)
        if not cle_s3:
            print(f"  bande absente : {band} ({code})")
            continue
        vsi_path = f"/vsis3/{CDSE_BUCKET}/{cle_s3}"
        dest = scene_dir / f"{band}.tif"
        if band == BANDE_REFERENCE:
            if dest.exists():
                with rasterio.open(dest) as src:
                    ref_profile = src.profile.copy()
                print(f"  déjà présent (référence) : {dest.name}")
            else:
                ref_profile = lire_fenetre_reference(vsi_path, dest)
                if ref_profile is None:
                    return False
        else:
            reprojeter_sur_reference(vsi_path, dest, ref_profile, band in BANDES_CATEGORIELLES)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2023-06-01")
    parser.add_argument("--end", default="2023-10-31")
    parser.add_argument("--max-cloud", type=float, default=30.0)
    parser.add_argument("--limit", type=int, default=4)
    args = parser.parse_args()

    cle, secret = verifier_identifiants()
    s3 = client_s3(cle, secret)

    print(f"Recherche Sentinel-2 L2A sur Copernicus Data Space ({args.start} -> {args.end})...")
    scenes = search_scenes(args.start, args.end, args.max_cloud, args.limit)
    print(f"{len(scenes)} scène(s) retenue(s).")

    index_path = RAW / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else []
    vus = {s["scene_id"] for s in index}

    for scene in scenes:
        scene_id = scene["scene_id"]
        if scene_id in vus:
            print(f"- {scene_id} déjà présent, ignoré")
            continue
        print(f"- {scene_id} ({scene['date']}, nuages {scene['cloud_cover']:.1f}%)")
        scene_dir = RAW / scene_id
        if download_scene_bands(s3, scene, scene_dir):
            index.append({"scene_id": scene_id, "date": scene["date"], "cloud_cover": scene["cloud_cover"], "dir": str(scene_dir)})
        elif scene_dir.exists():
            shutil.rmtree(scene_dir)

    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    print(f"\nIndex écrit dans {index_path} ({len(index)} scène(s) au total)")


if __name__ == "__main__":
    main()
