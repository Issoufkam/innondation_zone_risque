"""Télécharge les limites des 10 communes d'Abidjan depuis OpenStreetMap
(relations admin_level=8). Les limites OCHA (COD-AB) ne descendent que
jusqu'au département "Abidjan" (une seule zone) : OSM est la seule source
publique trouvée avec les communes séparées.

Identifiants de relation OSM (vérifiés manuellement le 07/11/2026,
Overpass "out tags" sur la zone d'étude) :
Treichville 3308305, Le Plateau 3308306, Marcory 3308780, Cocody 3363950,
Koumassi 3377741, Attécoubé 3377820, Adjamé 3377835, Port-Bouët 3377873,
Yopougon 3377881, Abobo 3377895.
"""

import json

import requests
from shapely.geometry import LineString, mapping
from shapely.ops import polygonize, unary_union

from config import BOUNDARIES_DIR

OVERPASS_SERVERS = [
    "https://overpass-api.de/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]
EN_TETES = {"User-Agent": "TribuneFloodSegmentation/1.0 (projet personnel, contact via GitHub)"}

RELATIONS_COMMUNES = {
    3308305: "Treichville",
    3308306: "Le Plateau",
    3308780: "Marcory",
    3363950: "Cocody",
    3377741: "Koumassi",
    3377820: "Attécoubé",
    3377835: "Adjamé",
    3377873: "Port-Bouët",
    3377881: "Yopougon",
    3377895: "Abobo",
}


def fetch_relations(ids):
    ids_str = ",".join(str(i) for i in ids)
    query = f"[out:json][timeout:90];relation(id:{ids_str});out geom;"
    last_error = None
    for server in OVERPASS_SERVERS:
        try:
            print(f"Interrogation de {server} ...")
            resp = requests.post(server, data={"data": query}, headers=EN_TETES, timeout=120)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:  # noqa: BLE001
            print(f"  échec ({exc}), serveur suivant...")
            last_error = exc
    raise RuntimeError(f"Tous les serveurs Overpass ont échoué : {last_error}")


def assembler_polygone(relation):
    """Reconstruit le(s) polygone(s) d'une relation admin à partir des
    géométries de ses membres "outer"/"inner" (polygonize sur les lignes)."""
    lignes_ext, lignes_int = [], []
    for membre in relation.get("members", []):
        if membre.get("type") != "way" or "geometry" not in membre:
            continue
        coords = [(pt["lon"], pt["lat"]) for pt in membre["geometry"]]
        if len(coords) < 2:
            continue
        ligne = LineString(coords)
        if membre.get("role") == "inner":
            lignes_int.append(ligne)
        else:
            lignes_ext.append(ligne)

    exterieur = unary_union(list(polygonize(lignes_ext)))
    if exterieur.is_empty:
        return None
    if lignes_int:
        interieur = unary_union(list(polygonize(lignes_int)))
        exterieur = exterieur.difference(interieur)
    return exterieur


def main():
    dest = BOUNDARIES_DIR / "communes_abidjan.geojson"
    if dest.exists():
        print(f"déjà présent : {dest}")
        return

    data = fetch_relations(RELATIONS_COMMUNES.keys())
    features = []
    for rel in data.get("elements", []):
        if rel.get("type") != "relation":
            continue
        nom = RELATIONS_COMMUNES.get(rel["id"], rel.get("tags", {}).get("name", "?"))
        geom = assembler_polygone(rel)
        if geom is None or geom.is_empty:
            print(f"  (géométrie vide, ignorée) {nom}")
            continue
        features.append(
            {
                "type": "Feature",
                "properties": {"commune": nom, "osm_relation_id": rel["id"]},
                "geometry": mapping(geom),
            }
        )
        print(f"  {nom} : ok")

    geojson = {"type": "FeatureCollection", "features": features}
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False)
    print(f"\n{len(features)} commune(s) écrite(s) dans {dest}")
    print("Source : OpenStreetMap contributors (licence ODbL) — à citer.")


if __name__ == "__main__":
    main()
