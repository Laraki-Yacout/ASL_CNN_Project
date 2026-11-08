# ASL_CNN_Project

**Classification d'images de la langue des signes américaine (ASL) à l'aide d'un réseau de neurones convolutif (CNN)**
*American Sign Language Image Classification Using Convolutional Neural Networks*

Mini-projet de Master ISI — module « Intelligence Artificielle et Innovation ».

## Objectif

Classifier une image 28×28 en niveaux de gris représentant une lettre statique de l'alphabet ASL parmi 24 classes (J et Z exclues car elles nécessitent un mouvement).

Le projet porte sur la classification d'images individuelles, et non sur la traduction continue de la langue des signes.

## Contenu du dépôt

| Élément | Rôle |
|---|---|
| `ASL_CNN_Classification.ipynb` | Notebook principal (Google Colab) |
| `references.md` | Sources du dataset et des documentations |
| `model/` | Modèle entraîné (`.keras`) et fichiers de classes |
| `results/` | Graphiques et métriques obtenus |

## Exécution

1. Ouvrir Google Colab → **Fichier → Ouvrir un notebook → GitHub**.
2. Coller l'adresse de ce dépôt et ouvrir `ASL_CNN_Classification.ipynb`.
3. Activer le GPU : **Exécution → Modifier le type d'exécution → GPU**.
4. Exécuter les cellules dans l'ordre.

## Outils

Python · TensorFlow / Keras · Google Colab

## Répartition

| Partie | Responsable |
|---|---|
| Sections 2 à 6 : configuration, données, exploration, prétraitement, architecture CNN | Étudiante 1 |
| Sections 7 à 12 : compilation, entraînement, évaluation, analyse des erreurs, prédiction, sauvegarde | Étudiante 2 |
| Introduction, conclusion, limites, vérification finale | Commun |
