# Segmentation sémantique des zones à risque d'inondation (Sentinel-2)

Projet data/IA : cartographier automatiquement les zones bâties exposées aux
risques d'inondation pour les collectivités territoriales, à partir de
l'imagerie satellite Sentinel-2 (bande visible/infrarouge/SWIR), des
empreintes de bâtiments Open Buildings et d'un modèle de segmentation
sémantique (U-Net).

Zone d'étude par défaut : agglomération d'Abidjan (Côte d'Ivoire).

## Sources de données (réelles, publiques, sans compte)

- **Imagerie Sentinel-2 L2A** : catalogue STAC public de
  [Digital Earth Africa](https://explorer.digitalearth.africa/stac)
  (collection `s2_l2a_c1`), fichiers hébergés sur un bucket S3 public
  (`deafrica-sentinel-2-l2a-c1`, région `af-south-1`), accessibles en HTTPS
  anonyme — aucune inscription Copernicus nécessaire.
- **Empreintes de bâtiments** : [Google Open Buildings v3](https://sites.research.google/gr/open-buildings/)
  (polygones), bucket public `open-buildings-data`, téléchargement par
  tuiles S2 (niveau 4) calculées avec `s2sphere`.
- **Limites administratives** : [Humanitarian Data Exchange — Côte d'Ivoire](https://data.humdata.org/group/civ)
  (COD-AB, limites des communes/régions).

Aucune valeur chiffrée du projet n'est inventée par une IA : tout vient de
ces fichiers téléchargés et des calculs du code.

## Méthode

1. `src/download_sentinel2.py` — recherche et télécharge les bandes
   Sentinel-2 (green, nir, swir16, swir22, scl) sur la zone et la période
   choisies.
2. `src/download_buildings.py` — télécharge les polygones Open Buildings
   qui intersectent la zone.
3. `src/download_boundaries.py` — télécharge les limites des communes
   (HDX).
4. `src/indices.py` — calcule le MNDWI (eau) et le NDBI (bâti) comme
   heuristiques de référence (baselines).
5. `src/labels.py` — construit les masques d'entraînement : classe
   « bâti » à partir des polygones Open Buildings (donnée vérifiée),
   classe « eau / zone inondée » à partir d'un seuillage MNDWI (heuristique
   — **limite méthodologique à mentionner** : ce n'est pas une carte
   d'inondation validée sur le terrain).
6. `src/tiling.py` — découpe la scène en vignettes (256×256) et sépare
   spatialement train/val/test (pas de chevauchement).
7. `src/unet.py` + `src/train.py` — entraîne un U-Net léger (CPU) pour la
   segmentation 3 classes (fond, bâti, eau).
8. `src/evaluate.py` — calcule IoU et F1-score par classe, exporte les
   masques prédits en GeoTIFF.

## Résultats (9 scènes Sentinel-2, 2021-2023, U-Net léger 25 époques)

Données réelles : 9 scènes Sentinel-2 L2A sur Abidjan (2021-2023), 235
vignettes d'entraînement / 53 validation / 56 test (256×256, séparation
spatiale stricte). Évaluation sur le jeu de test (`outputs/reports/evaluation.json`) :

| Classe              | IoU   | F1    |
|---------------------|-------|-------|
| Fond                | 0.85  | 0.92  |
| Bâti                | 0.13  | 0.24  |
| Eau / inondation    | 0.92  | 0.96  |

**La détection d'eau/zone inondée est solide (IoU 0,92).** La segmentation
du bâti est en revanche faible et **ne doit pas être présentée comme
fiable** : à 10 m de résolution, une parcelle résidentielle typique
d'Abidjan (quelques dizaines de m²) tient dans une fraction de pixel, ce
qui rend la classe « bâti » intrinsèquement difficile à apprendre à partir
des seules bandes optiques Sentinel-2. L'inférence sur une scène complète
(`carte_exposition.py`) confirme une forte sur-prédiction (~20% de bâti
prédit contre ~7% dans les étiquettes de référence).

Pistes pour améliorer la classe bâti : imagerie à plus haute résolution
(PlanetScope 3 m, imagerie commerciale très haute résolution), ou utiliser
directement les polygones Open Buildings/OSM comme sortie plutôt que de
les ré-apprendre depuis Sentinel-2 — Sentinel-2 est mieux adapté à la
détection de l'eau et de la végétation qu'à l'empreinte bâtie fine.

L'indicateur d'exposition (bâti à moins de 50 m d'une zone d'eau détectée,
`src/carte_exposition.py`) est calculé mais hérite de cette limite : à
publier uniquement avec la réserve ci-dessus, ou après une amélioration de
la classe bâti.

## Tableau de bord

- **Dashboard web** : [Segmentation Sentinel-2 — Abidjan](https://claude.ai/artifact/JYm4CUvAFTmDEdMiN3Czdj) (lien privé, à partager depuis le menu « Partager » de la page).
- **Kit Power BI** : `python src/kit_powerbi.py` écrit `outputs/kit_powerbi_segmentation_sentinel2.zip` (données CSV, thème, mesures DAX de départ — instructions dans son `LISEZMOI.txt`).
- `python src/resume.py` régénère les CSV consolidés (`outputs/reports/scenes.csv`, `metriques.csv`, `historique_entrainement.csv`, `exposition.csv`) utilisés par les deux.

## Limite de calcul

Pas de GPU Nvidia sur cette machine (carte graphique Intel Iris Xe) :
l'entraînement se fait sur CPU. Le pipeline est donc dimensionné pour
rester raisonnable en soirée : petite zone (Abidjan), peu de scènes,
U-Net réduit (peu de canaux), peu d'époques. Pour aller plus loin
(zone plus grande, plus d'époques), utiliser un notebook Google Colab
gratuit (GPU) — le code de `src/` est réutilisable tel quel.

## Installation

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Utilisation

```bash
python src/download_sentinel2.py --start 2023-06-01 --end 2023-10-31
python src/download_buildings.py
python src/download_boundaries.py
python src/indices.py
python src/labels.py
python src/tiling.py
python src/train.py --epochs 15
python src/evaluate.py
python src/visualiser.py          # aperçu PNG image/étiquette/prédiction (vignettes de test)
python src/carte_exposition.py    # inférence sur une scène complète + indicateurs d'exposition
```

Chaque étape écrit ses résultats dans `data/` ou `outputs/` et peut être
relancée indépendamment (les fichiers déjà téléchargés ne sont pas
retéléchargés).
