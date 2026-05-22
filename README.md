# Mini-Projet — When Machine Learning Fails

**Centrale Casablanca — ECC Spring 2026**  
*Introduction to AI and Machine Learning*  
---

## Vue d'ensemble

Ce dépôt contient l'investigation de deux modes d'échec d'un modèle non-linéaire
entraîné sur le dataset **Paddy (UCI 1186)** — données agronomiques et climatiques
de production rizicole dans le Tamil Nadu (Inde du Sud).

Le projet suit la posture scientifique demandée par l'énoncé :

> \"Force a model to fail, understand why it fails, and repair it."\

\---

## Structure du dépôt

```
mini_project_when_ml_fails/
│
├── data2.csv                              # Dataset Paddy (UCI 1186) — à placer ici
│
├── Reference_model.ipynb               # Modèle de référence (baseline)
├── Failure_distribution_shift.ipynb      # Mode d'échec principal
├── Failure_class_imbalance.ipynb         # Mode d'échec secondaire
│
├── requirements.txt                      # Dépendances Python
├── README.md                             # Ce fichier
└── rapport.pdf                           # Rapport écrit du projet
```

\---

## Ordre d'exécution recommandé

|Étape|Notebook|Rôle|
|-|-|-|
|**1**|`Reference_model.ipynb`|Établir les performances de référence sur les deux tâches|
|**2**|`Failure_distribution_shift.ipynb`|Investiguer le distribution shift géographique|
|**3**|`Failure_class_imbalance.ipynb`|Investiguer le déséquilibre de classes|

Chaque notebook est **entièrement indépendant** et peut être exécuté seul.

\---

## Dataset

|Propriété|Valeur|
|-|-|
|Source|UCI Machine Learning Repository — Dataset #1186|
|Taille|2 338 lignes × 45 colonnes (après déduplication)|
|Features numériques|36 (agronomiques + climatiques)|
|Features catégorielles|7 (Agriblock, Soil Types, Nursery, Wind Directions)|
|Tâche régression|Prédire le rendement rizicole (`Paddy yield(in Kg)`)|
|Tâche classification|Identifier la variété de riz (`Variety`)|
|Zones géographiques|6 Agriblocks (Tamil Nadu, Inde du Sud)|
|Classes classification|CO\_43 (26.1%), delux ponni (35.6%), ponmani (38.3%)|
|Valeurs manquantes|0|

\---

## Notebook 0 — Modèle de Référence

**Fichier :** `Reference_model.ipynb`

Ce notebook établit les **performances de référence** sur les deux tâches
(régression + classification) avec un pipeline propre et minimal.
Il sert de point de départ commun aux deux investigations de failure modes.

### Résultats de référence

|Tâche|Modèle|Métrique principale|
|-|-|-|
|Régression|`GradientBoostingRegressor`|R² = **0.9904** — RMSE = **923.0 Kg**|
|Classification|`RandomForestClassifier`|Accuracy = **0.9893** — F1-macro = **0.9888**|

### Pour exécuter ce notebook

```bash
jupyter notebook Reference_model.ipynb
```

\---

## Mode d'échec principal — Distribution Shift Géographique

**Fichier :** `Failure_distribution_shift.ipynb`

### Question de recherche

> Lorsqu'un GBR est entraîné sur un sous-ensemble de zones géographiques et
> déployé sur des zones entièrement nouvelles, la métrique agrégée (R²) cache-t-elle
> une dégradation systématique et statistiquement significative du RMSE sur ces
> zones non vues ?

### Résultats clés

|Scénario|RMSE moyen|Std inter-blocs|R² moyen|
|-|-|-|-|
|Modèle référence (split aléatoire)|873.6 Kg|—|0.9916|
|LOBO-CV — modèle brisé (avec Agriblock)|875.9 Kg|±76.5 Kg|0.9910|
|Fix 1 — Sans Agriblock, avec climat|876.8 Kg|**±69.7 Kg**|0.9909|
|Fix 2 — Modèle purement agronomique|886.2 Kg|±98.3 Kg|—|

**Conclusion :** H1 réfutée par test bootstrap CI 95% = \[815, 935].
La dégradation LOBO vs référence (+2.3 Kg) n'est pas statistiquement significative.
Cependant, la variance inter-blocs (RMSE 767–1 100 Kg, écart de **43%**) constitue
un risque opérationnel réel en déploiement sur de nouvelles zones géographiques.
Fix 1 réduit cette variance de **8.9%** en supprimant l'artefact d'encodage d'Agriblock.

### Pour exécuter ce notebook

```bash
jupyter notebook Failure_distribution_shift.ipynb
```

\---

## Mode d'échec secondaire — Déséquilibre de Classes

**Fichier :** `Failure_class_imbalance.ipynb`

### Question de recherche

> Lorsqu'un Random Forest est entraîné sur des données où `CO_43` est
> sous-représenté à un ratio croissant, son recall s'effondre-t-il de façon
> monotone tandis que l'accuracy reste artificiellement haute, et peut-on
> récupérer ce recall via un ajustement du seuil de décision sans ré-entraînement ?

### Résultats clés

|Modèle|Accuracy|F1-macro|BalAcc|Recall CO\_43|
|-|-|-|-|-|
|Référence (ratio 1.46:1)|0.9758|0.9738|0.9740|**0.9508**|
|Brisé (ratio 10.1:1)|0.8433|0.7995|0.8025|**0.4262** ← collapse|
|C1 — SMOTE|0.8533|0.8157|0.8152|0.4645|
|C2 — Seuil 0.10|0.9444|**0.9389**|0.9337|**0.8361**|
|C3 — SMOTE + Seuil|**0.9644**|**0.9620**|0.9595|**0.9290**|

**Conclusion :** H1 confirmée — le recall de CO\_43 décroît strictement :
0.95 → 0.92 → 0.87 → 0.72 → 0.43 → 0.31. La cause identifiée est le biais de
calibration des probabilités (P(CO\_43) médiane : 0.985 → 0.355 dans le modèle brisé).
L'ajustement du seuil 0.50 → 0.10 récupère **+42 points de recall** sans ré-entraînement.

### Pour exécuter ce notebook

```bash
jupyter notebook Failure_class_imbalance.ipynb
```

\---

## Reproductibilité

|Paramètre|Valeur|
|-|-|
|`RANDOM\_SEED`|`42` (défini en Section 0 de chaque notebook)|
|Frameworks|scikit-learn, imbalanced-learn (versions dans `requirements.txt`)|
|Pipelines|Modèle brisé ET modèle corrigé exécutables indépendamment|
|Preprocessing|Fitté **uniquement** sur le train set dans chaque notebook|

\---

## Modèles utilisés

|Notebook|Modèle|Famille|Tâche|
|-|-|-|-|
|`Reference_model.ipynb`|`GradientBoostingRegressor` + `RandomForestClassifier`|Tree-based ensembles|Régression + Classification|
|`Failure_distribution_shift.ipynb`|`GradientBoostingRegressor`|Tree-based ensemble|Régression|
|`Failure_class_imbalance.ipynb`|`RandomForestClassifier`|Tree-based ensemble|Classification|

\---

## Références

* Paddy dataset — UCI Machine Learning Repository, Dataset #1186.
https://archive.ics.uci.edu/dataset/1186
* Chawla et al. (2002). *SMOTE: Synthetic Minority Over-sampling Technique.*
Journal of Artificial Intelligence Research, 16, 321–357.
* Friedman, J. H. (2001). *Greedy function approximation: a gradient boosting machine.*
Annals of Statistics, 29(5), 1189–1232.
* Breiman, L. (2001). *Random forests.* Machine Learning, 45(1), 5–32.
* Pedregosa et al. (2011). *Scikit-learn: Machine Learning in Python.*
Journal of Machine Learning Research, 12, 2825–2830.
* Lemaître et al. (2017). *Imbalanced-learn: A Python Toolbox to Tackle the
Curse of Imbalanced Datasets.* JMLR, 18(17), 1–5.

