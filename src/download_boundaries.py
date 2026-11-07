"""Télécharge les limites administratives de la Côte d'Ivoire (communes,
régions) depuis le Humanitarian Data Exchange (OCHA COD-AB).

Source : https://data.humdata.org/dataset/cod-ab-civ (licence CC-BY-IGO).
"""

import io
import zipfile

import requests

from config import BOUNDARIES_DIR

URL = (
    "https://data.humdata.org/dataset/503a9167-ba56-4174-b160-bdb0fb703719/"
    "resource/321e55d2-3486-400a-a855-4b749cf9ef88/download/"
    "civ_admin_boundaries.geojson.zip"
)


def main():
    if any(BOUNDARIES_DIR.glob("*.geojson")):
        print(f"déjà présent dans {BOUNDARIES_DIR}")
        return
    print(f"Téléchargement : {URL}")
    resp = requests.get(URL, timeout=120)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        zf.extractall(BOUNDARIES_DIR)
    print(f"Limites administratives extraites dans {BOUNDARIES_DIR}")
    print("Source : OCHA / Côte d'Ivoire COD-AB (data.humdata.org) — à citer.")


if __name__ == "__main__":
    main()
