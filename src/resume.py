"""Consolide les résultats du projet (une scène, un modèle) en tables
prêtes pour un tableau de bord : statistiques par scène, métriques
d'évaluation, historique d'entraînement, indicateurs d'exposition.
Écrit des CSV dans outputs/reports/ (pour Power BI ou tout autre outil BI).
"""

import csv
import json
from pathlib import Path

import numpy as np
import rasterio

from config import PROCESSED, RAW, REPORTS_DIR

IGNORE_LABEL = 255


def stats_par_scene():
    index = json.loads((RAW / "index.json").read_text(encoding="utf-8"))
    manifest = json.loads((PROCESSED / "tiles" / "manifest.json").read_text(encoding="utf-8"))

    comptage_tiles = {}
    for split, items in manifest.items():
        for rel in items:
            scene_id = Path(rel).stem.rsplit("_r", 1)[0]
            comptage_tiles.setdefault(scene_id, {"train": 0, "val": 0, "test": 0})
            comptage_tiles[scene_id][split] += 1

    lignes = []
    for scene in index:
        scene_id = scene["scene_id"]
        with rasterio.open(PROCESSED / scene_id / "label.tif") as src:
            label = src.read(1)
        valide = label != IGNORE_LABEL
        total = valide.sum()
        tiles = comptage_tiles.get(scene_id, {"train": 0, "val": 0, "test": 0})
        lignes.append(
            {
                "scene_id": scene_id,
                "date": scene["date"],
                "nebulosite_tuile_pct": round(scene["cloud_cover"], 2),
                "pixels_invalides_aoi_pct": round(100 * (1 - valide.mean()), 2),
                "bati_pct": round(100 * (label == 1).sum() / max(total, 1), 2),
                "eau_inondation_pct": round(100 * (label == 2).sum() / max(total, 1), 2),
                "vignettes_train": tiles["train"],
                "vignettes_val": tiles["val"],
                "vignettes_test": tiles["test"],
            }
        )
    return lignes


def ecrire_csv(nom, lignes):
    if not lignes:
        return
    path = REPORTS_DIR / nom
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(lignes[0].keys()))
        writer.writeheader()
        writer.writerows(lignes)
    print(f"écrit : {path}")


def main():
    ecrire_csv("scenes.csv", stats_par_scene())

    evaluation = json.loads((REPORTS_DIR / "evaluation.json").read_text(encoding="utf-8"))
    lignes_eval = [
        {"classe": classe, "iou": round(v["iou"], 4), "f1": round(v["f1"], 4)}
        for classe, v in evaluation.items()
    ]
    ecrire_csv("metriques.csv", lignes_eval)

    historique = json.loads((REPORTS_DIR / "historique_entrainement.json").read_text(encoding="utf-8"))
    ecrire_csv("historique_entrainement.csv", historique)

    if (REPORTS_DIR / "exposition.json").exists():
        exposition = json.loads((REPORTS_DIR / "exposition.json").read_text(encoding="utf-8"))
        ecrire_csv("exposition.csv", [exposition])

    print("\nTerminé.")


if __name__ == "__main__":
    main()
