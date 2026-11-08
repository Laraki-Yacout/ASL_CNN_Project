# Classification d'images de la langue des signes américaine (ASL) avec un CNN

*American Sign Language Image Classification Using Convolutional Neural Networks*

[![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Laraki-Yacout/ASL_CNN_Project/blob/main/ASL_CNN_Classification.ipynb)

Projet de **Master BD2C**, module « Intelligence Artificielle et Innovation ».
Réseau de neurones convolutif entraîné à reconnaître les **24 lettres statiques de l'alphabet ASL** à partir d'images 28 × 28 en niveaux de gris.

---

## Objectif

À partir d'une image d'une main, prédire la lettre ASL représentée et afficher les probabilités des 24 classes.

Le système :

1. reçoit une image ;
2. la prétraite (niveaux de gris, 28 × 28, normalisation) ;
3. extrait automatiquement ses caractéristiques visuelles avec des couches de convolution ;
4. prédit la lettre et affiche les scores ;
5. est évalué sur un jeu de test indépendant.

> Le projet classe des **images individuelles**. Il ne traduit pas de mots ni de phrases en langue des signes.

## Dataset

**[Sign Language MNIST](https://www.kaggle.com/datasets/datamunge/sign-language-mnist)** (Kaggle, licence CC0)

| Caractéristique | Valeur vérifiée dans le notebook |
|---|---|
| Images d'entraînement | 27 455 |
| Images de test | 7 172 |
| Taille | 28 × 28, niveaux de gris (pixels 0–255) |
| Classes | 24 lettres (A–Y sans J ; Z absent) |
| Format | CSV : 1 colonne `label` + 784 colonnes de pixels |
| Valeurs manquantes | aucune |

J et Z sont exclues car elles nécessitent un mouvement. Le dataset a été construit à partir de **1 704 photos d'origine**, chacune déclinée en de nombreuses variantes (recadrage, filtres, bruit, luminosité, rotation).

## Méthode

1. **Chargement** avec KaggleHub et vérification des fichiers.
2. **Exploration** : répartition des classes, exemples d'images, distribution des pixels.
3. **Prétraitement** :
   - suppression des doublons exacts dans le train et des images du train identiques au test ;
   - réencodage des labels (0–24 sans 9) en indices consécutifs 0–23 ;
   - mise en forme 28 × 28 × 1 et normalisation entre 0 et 1 ;
   - séparation train / validation stratifiée (80 / 20), test officiel conservé à part ;
   - mesure de la proximité validation / test avec le train (risque de fuite lié aux variantes d'une même photo).
4. **Architecture CNN** (112 280 paramètres) :

   ```
   Input 28×28×1
   → Conv2D 32 (3×3, ReLU) → MaxPooling 2×2
   → Conv2D 64 (3×3, ReLU) → MaxPooling 2×2
   → Conv2D 128 (3×3, ReLU) → MaxPooling 2×2
   → GlobalAveragePooling2D
   → Dense 128 (ReLU) → Dropout 0,3
   → Dense 24 (Softmax)
   ```

5. **Entraînement** : Adam (lr = 0,001), `SparseCategoricalCrossentropy`, batch 64, EarlyStopping sur la perte de validation. Trois volumes de données comparés sur la validation : 300 images/classe, 600 images/classe, tout le train.
6. **Évaluation unique sur le test** : accuracy, précision, rappel, F1, matrice de confusion, analyse des erreurs.
7. **Prédiction sur une image externe** et **sauvegarde** du modèle (`.keras`) avec l'ordre des classes.

## Résultats

Exécution du 8 novembre 2026 sur Google Colab (GPU T4).

### Performance sur le test (7 172 images, évaluation unique)

| Métrique | Valeur |
|---|---|
| **Accuracy** | **96,17 %** |
| F1 macro | 0,960 |
| Précision macro | 0,957 |
| Rappel macro | 0,965 |
| Erreurs | 275 / 7 172 |

### Effet du volume de données (validation)

| Données d'entraînement | Meilleure époque | Accuracy validation | Loss validation |
|---|---|---|---|
| 300 images / classe | 35 | 99,85 % | 0,0122 |
| 600 images / classe | 24 | 99,98 % | 0,0016 |
| Tout le train (21 964) | 21 | 100 % | 0,0003 |

![Courbes d'apprentissage](results/courbes_apprentissage.png)

### Validation optimiste : un point méthodologique

La validation atteint 100 % mais le test 96,17 %. Une mesure de distance au plus proche voisin montre que les images de validation sont des **variantes très proches** d'images du train (même photo d'origine), contrairement aux images de test. Le test, gardé intact jusqu'à la fin, est donc la seule estimation fiable.

![Proximité validation / test](results/proximite_val_test.png)

### Matrice de confusion

![Matrice de confusion](results/matrice_confusion.png)

- F, L et O sont reconnues parfaitement (F1 = 1,00).
- Les lettres les plus difficiles sont R (F1 0,883), G (0,885) et W (0,905).
- **84 % des erreurs** concernent 10 paires de signes visuellement proches : B→W (41), M→S (31), Y→I (27), K→R (21), N→A (21), H→G (20), T→G (20)…

![Principales confusions](results/principales_confusions.png)

## Limites

- Les images sont très homogènes (main recadrée, 28 × 28, peu de personnes et d'arrière-plans) : les performances ne garantissent pas une reconnaissance fiable sur des photos prises en conditions réelles.
- Le dataset provient de 1 704 photos seulement ; des variantes d'une même photo peuvent se trouver à la fois dans le train et la validation, ce qui peut rendre la validation optimiste.
- J et Z (lettres en mouvement), les mots et les phrases ne sont pas couverts.

## Perspectives

- Augmentation de données (rotations, décalages, luminosité) pour améliorer la généralisation.
- Test sur des photos réelles ou un autre dataset ASL.
- Résolution plus élevée pour distinguer les détails fins (position du pouce).
- Explicabilité (Grad-CAM) pour vérifier que le modèle regarde la main et non l'arrière-plan.

## Structure du dépôt

```
ASL_CNN_Project/
├── ASL_CNN_Classification.ipynb   # notebook complet (Colab)
├── README.md
├── references.md                   # sources
├── model/                          # modèle entraîné (.keras) + classes.json
└── results/                        # graphiques et métriques
```

## Exécution

Le modèle entraîné est fourni dans `model/` : on peut le recharger avec `keras.models.load_model("model/asl_cnn.keras")` et l'ordre des classes avec `model/classes.json`.

Pour tout réexécuter :

1. Cliquer sur le badge **Open in Colab** ci-dessus.
2. **Exécution → Modifier le type d'exécution → T4 GPU**.
3. **Exécution → Tout exécuter**.

## Technologies

Python · TensorFlow 2.21 · Keras 3 · NumPy · Pandas · scikit-learn · Matplotlib · Google Colab

## Auteures

Projet réalisé en binôme dans le cadre du Master BD2C.
