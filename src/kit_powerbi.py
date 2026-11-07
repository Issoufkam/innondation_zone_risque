"""Assemble un kit Power BI (ZIP) : données (CSV), thème de rapport, et
mesures DAX de départ, prêt à importer dans Power BI Desktop.

Ne crée pas de fichier .pbix (Power BI Desktop n'est pas disponible dans
cet environnement) : ce kit contient tout ce qu'il faut pour construire
le rapport soi-même en quelques minutes (Obtenir les données > Dossier,
puis Affichage > Parcourir les thèmes > Rechercher les thèmes).
"""

import json
import shutil
import zipfile
from pathlib import Path

from config import REPORTS_DIR, ROOT

THEME = {
    "name": "Segmentation Sentinel-2",
    "dataColors": [
        "#1F6FB2",  # eau / inondation
        "#B2461F",  # bâti
        "#6B7280",  # fond / neutre
        "#2E9E6B",
        "#C98A1F",
        "#7A5FB0",
    ],
    "background": "#FFFFFF",
    "foreground": "#18222F",
    "tableAccent": "#1F6FB2",
    "good": "#2E9E6B",
    "neutral": "#C98A1F",
    "bad": "#B2461F",
    "visualStyles": {
        "*": {
            "*": {
                "*": [
                    {
                        "background": {"color": {"solid": {"color": "#FFFFFF"}}},
                        "border": {"show": True, "color": {"solid": {"color": "#E5E7EB"}}},
                    }
                ]
            }
        }
    },
}

LISEZMOI = """Kit Power BI — Segmentation sémantique Sentinel-2 (Abidjan)
============================================================

Contenu
--------
- donnees/*.csv              : scènes, métriques du modèle, historique
                                d'entraînement, indicateur d'exposition.
                                Aucun chiffre n'est inventé : tout vient
                                du pipeline (voir README.md du dépôt).
- theme.json                 : thème de rapport Power BI (couleurs).
- mesures_dax.txt            : mesures DAX de départ, à coller dans
                                Power BI après l'import des données.

Import dans Power BI Desktop
-----------------------------
1. Accueil > Obtenir les données > Dossier texte/CSV, sélectionner
   chaque fichier de donnees/ (ou "Obtenir les données > Dossier" en
   pointant sur donnees/ puis "Combiner et transformer" si vous préférez
   une seule requête).
2. Affichage > Thèmes > Parcourir les thèmes dans votre fichier,
   sélectionner theme.json.
3. Dans chaque table, coller les mesures de mesures_dax.txt (clic droit
   sur la table > Nouvelle mesure).
4. Visuels suggérés (reprend le gabarit du projet) :
   - Cartes (KPI) : IoU Bâti, IoU Eau, F1-Score global (moyenne de
     metriques[f1]), Nombre de scènes, Vignettes train
   - Graphique à barres groupées : exposition_communes (axe = commune,
     valeur = surface_bati_expose_ha, filtre = distance_risque_m)
   - Histogramme groupé : metriques (axe = classe, valeur = iou)
   - Graphique en courbes : historique_entrainement (axe = epoch,
     valeurs = iou_fond, iou_bati, iou_eau)
   - Table : scenes (date, bati_pct, eau_inondation_pct, vignettes_*)
   - Filtres (toutes les pages) : scenes[date] / exposition_communes[commune]
     et exposition_communes[distance_risque_m]

Limite à rappeler sur le rapport
----------------------------------
La classe « bâti » est sur-prédite par le modèle actuel (IoU 0,13) : à
10 m de résolution Sentinel-2, une parcelle résidentielle d'Abidjan tient
dans une fraction de pixel. L'indicateur d'exposition hérite de cette
limite — à afficher avec cette réserve sur le rapport, pas comme un
chiffre définitif.

Sources des données d'origine (à citer)
------------------------------------------
- Sentinel-2 L2A : Copernicus / Digital Earth Africa
- Bâtiments : OpenStreetMap contributors (licence ODbL)
- Limites administratives : OCHA Côte d'Ivoire (COD-AB)
"""

DAX = """// Mesures de départ — à coller dans Power BI après l'import des CSV.
// Table metriques : colonnes classe, iou, f1

IoU Eau = CALCULATE(SUM(metriques[iou]); metriques[classe] = "eau_inondation")
IoU Bati = CALCULATE(SUM(metriques[iou]); metriques[classe] = "bati")
IoU Fond = CALCULATE(SUM(metriques[iou]); metriques[classe] = "fond")
F1 Score Global = AVERAGE(metriques[f1])

// Table exposition_communes : colonnes commune, distance_risque_m,
// surface_bati_ha, surface_eau_ha, surface_bati_expose_ha,
// part_bati_expose_pct — filtrer par distance_risque_m (50, 100 ou 200)
// pour le graphique "bâti exposé par commune" (axe = commune)

Surface bati expose par commune (km2) = SUM(exposition_communes[surface_bati_expose_ha]) / 100

// Table scenes : colonnes date, pixels_invalides_aoi_pct, bati_pct,
// eau_inondation_pct, vignettes_train, vignettes_val, vignettes_test

Nombre de scenes = COUNTROWS(scenes)
Vignettes entrainement = SUM(scenes[vignettes_train])
Vignettes validation = SUM(scenes[vignettes_val])
Vignettes test = SUM(scenes[vignettes_test])
Nebulosite moyenne AOI = AVERAGE(scenes[pixels_invalides_aoi_pct])

// Table exposition : une seule ligne (scène de référence)

Surface bati totale (ha) = SUM(exposition[surface_bati_totale_ha])
Surface bati expose (ha) = SUM(exposition[surface_bati_expose_ha])
Part bati expose (%) = SUM(exposition[part_bati_expose_pct])

// Table historique_entrainement : colonnes epoch, perte_train, perte_val
// (utiliser directement comme axe/valeurs dans un graphique en courbes,
// aucune mesure n'est nécessaire)
"""


def main():
    kit_dir = ROOT / "outputs" / "powerbi_kit"
    donnees_dir = kit_dir / "donnees"
    donnees_dir.mkdir(parents=True, exist_ok=True)

    for nom in [
        "scenes.csv",
        "metriques.csv",
        "historique_entrainement.csv",
        "exposition.csv",
        "exposition_communes.csv",
    ]:
        src = REPORTS_DIR / nom
        if src.exists():
            shutil.copy(src, donnees_dir / nom)
        else:
            print(f"absent, ignoré : {nom}")

    (kit_dir / "theme.json").write_text(json.dumps(THEME, indent=2, ensure_ascii=False), encoding="utf-8")
    (kit_dir / "LISEZMOI.txt").write_text(LISEZMOI, encoding="utf-8", newline="\r\n")
    (kit_dir / "mesures_dax.txt").write_text(DAX, encoding="utf-8")

    zip_path = ROOT / "outputs" / "kit_powerbi_segmentation_sentinel2.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in kit_dir.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(kit_dir))

    print(f"Kit Power BI écrit dans {zip_path}")


if __name__ == "__main__":
    main()
