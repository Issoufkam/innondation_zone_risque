# Segmentation sémantique des zones à risque d'inondation (Sentinel-2)

Projet data/IA : cartographier automatiquement les zones bâties exposées aux
risques d'inondation pour les collectivités territoriales, à partir de
l'imagerie satellite Sentinel-2 (bande visible/infrarouge/SWIR), des
empreintes de bâtiments Open Buildings et d'un modèle de segmentation
sémantique (U-Net).

Zone d'étude par défaut : agglomération d'Abidjan (Côte d'Ivoire).

## Sources de données (réelles, publiques)

- **Imagerie Sentinel-2 L2A** — deux sources interchangeables (même
  pipeline derrière) :
  - [Digital Earth Africa](https://explorer.digitalearth.africa/stac)
    (collection `s2_l2a_c1`) : fichiers hébergés sur un bucket S3 public
    (`deafrica-sentinel-2-l2a-c1`, région `af-south-1`), accessibles en
    HTTPS anonyme — **aucun compte nécessaire**, source par défaut
    (`src/download_sentinel2.py`).
  - [Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu)
    (la source officielle citée dans le projet d'origine) : nécessite un
    compte et des clés S3 (`src/download_sentinel2_cdse.py`, voir
    `.env.example`).
- **Empreintes de bâtiments** : OpenStreetMap (relevé communautaire,
  licence ODbL), interrogé via l'API Overpass (`src/download_buildings.py`).
  Préféré à Google Open Buildings pour cette zone : la tuile S2 Open
  Buildings qui couvre Abidjan pèse 1,5 Go (pas de découpage plus fin sans
  compte Google Earth Engine), contre quelques Mo via OSM.
- **Communes d'Abidjan** : OpenStreetMap (relations `admin_level=8`,
  `src/download_communes.py`) — les limites OCHA (COD-AB) ne descendent
  que jusqu'au département "Abidjan" (une seule zone, pas les 10 communes).
- **Limites administratives (pays/régions)** : [Humanitarian Data Exchange — Côte d'Ivoire](https://data.humdata.org/group/civ)
  (COD-AB, `src/download_boundaries.py`).

Aucune valeur chiffrée du projet n'est inventée par une IA : tout vient de
ces fichiers téléchargés et des calculs du code.

## Méthode

1. `src/download_sentinel2.py` (ou `download_sentinel2_cdse.py`) —
   recherche et télécharge les bandes Sentinel-2 (green, nir, swir16,
   swir22, scl) sur la zone et la période choisies ; seule la fenêtre qui
   recouvre l'AOI est lue, jamais la scène entière.
2. `src/download_buildings.py` — télécharge les empreintes de bâtiments
   OpenStreetMap qui intersectent la zone.
3. `src/download_communes.py` — télécharge les 10 communes d'Abidjan
   (OpenStreetMap).
4. `src/download_boundaries.py` — télécharge les limites administratives
   du pays/des régions (HDX).
5. `src/indices.py` — calcule le MNDWI (eau) et le NDBI (bâti) comme
   heuristiques de référence (baselines).
6. `src/labels.py` — construit les masques d'entraînement : classe
   « bâti » à partir des empreintes OpenStreetMap (donnée vérifiée),
   classe « eau / zone inondée » à partir d'un seuillage MNDWI (heuristique
   — **limite méthodologique à mentionner** : ce n'est pas une carte
   d'inondation validée sur le terrain).
7. `src/tiling.py` — découpe la scène en vignettes (256×256) et sépare
   spatialement train/val/test (pas de chevauchement).
8. `src/unet.py` + `src/train.py` — entraîne un U-Net léger (CPU) pour la
   segmentation 3 classes (fond, bâti, eau) ; suit la perte et l'IoU de
   validation à chaque époque.
9. `src/evaluate.py` — calcule IoU et F1-score par classe sur le jeu de
   test, exporte les masques prédits en GeoTIFF.
10. `src/carte_exposition.py` — applique le modèle à une scène complète et
    calcule l'exposition du bâti (global et par commune, à 50/100/200 m
    d'une zone d'eau détectée).

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

Pour utiliser Copernicus Data Space au lieu de Digital Earth Africa :
copier `.env.example` en `.env` et renseigner `CDSE_S3_ACCESS_KEY` /
`CDSE_S3_SECRET_KEY` (clés générées sur
https://eodata-s3keysmanager.dataspace.copernicus.eu/, jamais commitées).

## Utilisation

```bash
python src/download_sentinel2.py --start 2023-06-01 --end 2023-10-31
# ou, avec un compte Copernicus : python src/download_sentinel2_cdse.py --start 2023-06-01 --end 2023-10-31
python src/download_buildings.py
python src/download_communes.py
python src/download_boundaries.py
python src/indices.py
python src/labels.py
python src/tiling.py
python src/train.py --epochs 25
python src/evaluate.py
python src/visualiser.py          # aperçu PNG image/étiquette/prédiction (vignettes de test)
python src/carte_exposition.py    # inférence sur une scène complète + exposition (globale et par commune)
python src/resume.py              # consolide les résultats en CSV (dashboard, Power BI)
python src/kit_powerbi.py         # assemble le kit Power BI (ZIP)
```

Chaque étape écrit ses résultats dans `data/` ou `outputs/` et peut être
relancée indépendamment (les fichiers déjà téléchargés ne sont pas
retéléchargés).
