"""Télécharge les empreintes de bâtiments OpenStreetMap (Overpass API) sur
la zone d'étude et les enregistre en GeoJSON.

Source : OpenStreetMap contributors, licence ODbL — à citer dans toute
publication. Interrogé via l'API Overpass publique (plusieurs serveurs,
avec repli en cas d'erreur).
"""

import json
import time

import requests

from config import AOI_BBOX, BUILDINGS_DIR

OVERPASS_SERVERS = [
    "https://overpass-api.de/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]

EN_TETES = {"User-Agent": "TribuneFloodSegmentation/1.0 (projet personnel, contact via GitHub)"}


def overpass_bbox(bbox):
    min_lon, min_lat, max_lon, max_lat = bbox
    return f"{min_lat},{min_lon},{max_lat},{max_lon}"


def fetch_buildings(bbox) -> dict:
    query = f"""
    [out:json][timeout:180];
    (
      way["building"]({overpass_bbox(bbox)});
    );
    out geom;
    """
    last_error = None
    for server in OVERPASS_SERVERS:
        for tentative in range(2):
            try:
                print(f"Interrogation de {server} (tentative {tentative + 1}) ...")
                resp = requests.post(server, data={"data": query}, headers=EN_TETES, timeout=190)
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:  # noqa: BLE001 - on réessaie / serveur suivant
                print(f"  échec ({exc})")
                last_error = exc
                time.sleep(5)
    raise RuntimeError(f"Tous les serveurs Overpass ont échoué : {last_error}")


def ways_to_geojson(data: dict) -> dict:
    features = []
    for el in data.get("elements", []):
        if el.get("type") != "way" or "geometry" not in el:
            continue
        coords = [[pt["lon"], pt["lat"]] for pt in el["geometry"]]
        if len(coords) < 4 or coords[0] != coords[-1]:
            continue  # pas un polygone fermé exploitable
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "osm_id": el["id"],
                    "building": el.get("tags", {}).get("building", "yes"),
                },
                "geometry": {"type": "Polygon", "coordinates": [coords]},
            }
        )
    return {"type": "FeatureCollection", "features": features}


def main():
    dest = BUILDINGS_DIR / "batiments_abidjan.geojson"
    if dest.exists():
        print(f"déjà présent : {dest}")
        return
    data = fetch_buildings(AOI_BBOX)
    geojson = ways_to_geojson(data)
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False)
    print(f"{len(geojson['features'])} bâtiments écrits dans {dest}")
    print("Source : OpenStreetMap contributors (licence ODbL) — à citer.")


if __name__ == "__main__":
    main()
