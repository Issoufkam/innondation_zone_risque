"""Paramètres communs du projet : zone d'étude, chemins, constantes."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _charger_env(path):
    """Lecteur minimal de fichier .env (évite une dépendance à python-dotenv)."""
    if not path.exists():
        return
    for ligne in path.read_text(encoding="utf-8").splitlines():
        ligne = ligne.strip()
        if not ligne or ligne.startswith("#") or "=" not in ligne:
            continue
        cle, _, valeur = ligne.partition("=")
        os.environ.setdefault(cle.strip(), valeur.strip())


_charger_env(ROOT / ".env")
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
BUILDINGS_DIR = DATA / "buildings"
BOUNDARIES_DIR = DATA / "boundaries"
OUTPUTS = ROOT / "outputs"
MASKS_DIR = OUTPUTS / "masks"
CHECKPOINTS_DIR = OUTPUTS / "checkpoints"
REPORTS_DIR = OUTPUTS / "reports"

for d in (RAW, PROCESSED, BUILDINGS_DIR, BOUNDARIES_DIR, MASKS_DIR, CHECKPOINTS_DIR, REPORTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Zone d'étude par défaut : agglomération d'Abidjan (Côte d'Ivoire).
# [min_lon, min_lat, max_lon, max_lat]
AOI_BBOX = (-4.15, 5.20, -3.85, 5.45)

DEA_STAC_URL = "https://explorer.digitalearth.africa/stac/search"
DEA_COLLECTION = "s2_l2a_c1"

# Bandes Sentinel-2 utiles : vert/NIR/SWIR pour MNDWI, NIR/SWIR pour NDBI.
S2_BANDS = ["green", "nir", "swir16", "swir22", "scl"]

# Classes des masques de segmentation
CLASS_BACKGROUND = 0
CLASS_BUILDING = 1
CLASS_WATER = 2
CLASS_NAMES = {0: "fond", 1: "bati", 2: "eau_inondation"}

TILE_SIZE = 256
