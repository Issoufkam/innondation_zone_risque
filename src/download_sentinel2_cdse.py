"""Télécharge les bandes Sentinel-2 L2A depuis la source officielle
Copernicus Data Space Ecosystem (CDSE), via son stockage compatible S3
("eodata"). Alternative à download_sentinel2.py (qui utilise Digital
Earth Africa, sans compte) : celle-ci utilise la source citée dans le
projet d'origine, mais nécessite un compte Copernicus et des clés S3.

Clés S3 : à générer sur https://eodata-s3keysmanager.dataspace.copernicus.eu/
puis à mettre dans .env (voir .env.example) — jamais commitées.

Particularité de cette zone d'étude : l'AOI d'Abidjan tombe exactement au
croisement de 4 tuiles MGRS standard (30NVM, 30NUM, 30NVL, 30NUL) — Digital
Earth Africa l'évite car son produit "Collection 1" est regrillé sur une
grille continentale propre, mais la donnée Copernicus officielle est
encore découpée selon le quadrillage satellite d'origine. Ce script
mosaïque donc les 4 tuiles (quand elles existent pour une date) sur une
grille de référence fixée une fois pour toutes à partir de l'AOI.
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import boto3
import numpy as np
import rasterio
import requests
from affine import Affine
from rasterio.enums import Resampling
from rasterio.warp import reproject, transform_bounds

from config import AOI_BBOX, RAW

CDSE_ODATA_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
CDSE_S3_ENDPOINT = "eodata.dataspace.copernicus.eu"
CDSE_BUCKET = "eodata"

# Les 4 tuiles MGRS standard qui se partagent l'AOI d'Abidjan (vérifié le
# 07/11/2026 par recherche OData sans filtre de tuile).
TUILES_MGRS = ["30NVM", "30NUM", "30NVL", "30NUL"]

# CRS cible de la grille de référence : UTM 30N, qui couvre toute la zone
# d'étude (Abidjan, ~4°O) — comme les 4 tuiles MGRS ci-dessus.
CRS_CIBLE = "EPSG:32630"
RESOLUTION_M = 10

BANDES_CDSE = {"green": "B03", "nir": "B08", "swir16": "B11", "swir22": "B12", "scl": "SCL"}
DOSSIER_PAR_BANDE = {"B03": "R10m/", "B08": "R10m/", "B11": "R20m/", "B12": "R20m/", "SCL": "R20m/"}
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


def grille_reference():
    """Grille 10 m fixée une fois pour toutes à partir de l'AOI (indépendante
    de toute tuile), sur laquelle les 4 tuiles sont mosaïquées."""
    min_x, min_y, max_x, max_y = transform_bounds("EPSG:4326", CRS_CIBLE, *AOI_BBOX)
    largeur = round((max_x - min_x) / RESOLUTION_M)
    hauteur = round((max_y - min_y) / RESOLUTION_M)
    transform = Affine(RESOLUTION_M, 0, min_x, 0, -RESOLUTION_M, max_y)
    return largeur, hauteur, transform


def search_dates(start: str, end: str, max_cloud: float, limit: int):
    """Recherche groupée par date : renvoie, pour chaque date retenue, les
    produits des tuiles trouvées (jusqu'à 4)."""
    min_lon, min_lat, max_lon, max_lat = AOI_BBOX
    polygon = (
        f"POLYGON(({min_lon} {min_lat},{max_lon} {min_lat},{max_lon} {max_lat},"
        f"{min_lon} {max_lat},{min_lon} {min_lat}))"
    )
    filtre = (
        "Collection/Name eq 'SENTINEL-2' and contains(Name,'MSIL2A') and "
        f"OData.CSC.Intersects(area=geography'SRID=4326;{polygon}') and "
        f"ContentDate/Start gt {start}T00:00:00.000Z and ContentDate/Start lt {end}T23:59:59.000Z"
    )
    resp = requests.get(CDSE_ODATA_URL, params={"$filter": filtre, "$top": 100, "$expand": "Attributes"}, timeout=60)
    resp.raise_for_status()
    produits = resp.json().get("value", [])

    par_date = defaultdict(dict)
    for p in produits:
        attrs = {a["Name"]: a.get("Value") for a in p.get("Attributes", [])}
        tuile = attrs.get("tileId")
        if tuile not in TUILES_MGRS:
            continue
        date = p["ContentDate"]["Start"][:10]
        par_date[date][tuile] = {
            "cloud_cover": attrs.get("cloudCover", 100),
            "s3_path": p["S3Path"].lstrip("/"),
        }

    dates = []
    for date, tuiles in par_date.items():
        nebulosite = max(t["cloud_cover"] for t in tuiles.values())
        if nebulosite <= max_cloud:
            dates.append((date, nebulosite, tuiles))
    dates.sort(key=lambda x: x[1])
    return dates[:limit]


def trouver_fichiers_bandes(s3, s3_path):
    prefix_granule = s3_path.split("eodata/", 1)[-1] + "/GRANULE/"
    resp = s3.list_objects_v2(Bucket=CDSE_BUCKET, Prefix=prefix_granule, Delimiter="/")
    granules = [cp["Prefix"] for cp in resp.get("CommonPrefixes", [])]
    if not granules:
        return {}
    granule = granules[0]

    listages, fichiers = {}, {}
    for code, dossier in DOSSIER_PAR_BANDE.items():
        if dossier not in listages:
            resp = s3.list_objects_v2(Bucket=CDSE_BUCKET, Prefix=granule + "IMG_DATA/" + dossier)
            listages[dossier] = [obj["Key"] for obj in resp.get("Contents", [])]
        suffixe = f"_{code}_{dossier.rstrip('/').lstrip('R')}.jp2"
        for cle in listages[dossier]:
            if cle.endswith(suffixe):
                fichiers[code] = cle
                break
    return fichiers


def mosaiquer_bande(s3, tuiles: dict, code: str, categorique: bool, largeur, hauteur, transform):
    composite = np.zeros((hauteur, largeur), dtype="uint16")
    au_moins_une = False
    for tuile, info in tuiles.items():
        fichiers = trouver_fichiers_bandes(s3, info["s3_path"])
        cle = fichiers.get(code)
        if not cle:
            print(f"    tuile {tuile} : bande {code} absente")
            continue
        vsi_path = f"/vsis3/{CDSE_BUCKET}/{cle}"
        with rasterio.open(vsi_path) as src:
            destination = np.zeros((hauteur, largeur), dtype="uint16")
            reproject(
                source=rasterio.band(src, 1),
                destination=destination,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=CRS_CIBLE,
                resampling=Resampling.nearest if categorique else Resampling.bilinear,
            )
        composite = np.where(destination > 0, destination, composite)
        au_moins_une = True
    return composite if au_moins_une else None


def download_scene(s3, tuiles, scene_dir: Path):
    largeur, hauteur, transform = grille_reference()
    profile = {
        "driver": "GTiff",
        "height": hauteur,
        "width": largeur,
        "count": 1,
        "dtype": "uint16",
        "crs": CRS_CIBLE,
        "transform": transform,
    }
    scene_dir.mkdir(parents=True, exist_ok=True)
    for band, code in BANDES_CDSE.items():
        dest = scene_dir / f"{band}.tif"
        if dest.exists():
            print(f"  déjà présent : {dest.name}")
            continue
        composite = mosaiquer_bande(s3, tuiles, code, band in BANDES_CATEGORIELLES, largeur, hauteur, transform)
        if composite is None:
            print(f"  bande absente sur toutes les tuiles : {band}")
            continue
        with rasterio.open(dest, "w", **profile) as dst:
            dst.write(composite, 1)
        print(f"  mosaïque écrite : {dest.name} ({len(tuiles)} tuile(s))")
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
    dates = search_dates(args.start, args.end, args.max_cloud, args.limit)
    print(f"{len(dates)} date(s) retenue(s).")

    index_path = RAW / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else []
    vus = {s["scene_id"] for s in index}

    for date, nebulosite, tuiles in dates:
        scene_id = f"CDSE_{date.replace('-', '')}_T30N_mosaic"
        if scene_id in vus:
            print(f"- {scene_id} déjà présent, ignoré")
            continue
        print(f"- {scene_id} ({date}, nébulosité max {nebulosite:.1f}% sur {len(tuiles)} tuile(s) : {list(tuiles)})")
        scene_dir = RAW / scene_id
        if download_scene(s3, tuiles, scene_dir):
            index.append({"scene_id": scene_id, "date": date, "cloud_cover": nebulosite, "dir": str(scene_dir)})

    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    print(f"\nIndex écrit dans {index_path} ({len(index)} scène(s) au total)")


if __name__ == "__main__":
    main()
