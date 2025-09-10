# Projet ResNet MCDropOut

Ce dépôt contient l'implémentation d'un réseau de neurones ResNet50 avec Dropout Monte Carlo (MCDropOut) et plusieurs scripts pour l'analyse des données. Ce projet vise à appliquer cette architecture pour des tâches de classification d'images, avec des fonctionnalités d'analyse et de prétraitement des données.

## Structure du dépôt

- **data/**
- **models/**
- **runs/**
- **scripts/**
  - **`utils/`** : Contient des scripts utilitaires pour la gestion des données et l'entraînement du modèle.
    - **`dataloader_with_aug.py`** : Implémentation du DataLoader avec augmentation des données.
    - **`merge_dataframe_cluster.py`** : Fusion des clusters dans un DataFrame.
  - **`analyse_clusters.py`** : Outils pour analyser les clusters générés.
  - **`analyse.py`** : Scripts d'analyse des résultats du modèle.
  - **`inference.py`** : Effectue l'inférence sur un modèle pré-entrainé.
  - **`resnet50_mc_dropout.py`** : Implémentation du modèle ResNet50 avec dropout Monte Carlo.
  - **`train.py`** : Script principal pour l'entraînement du modèle.
- **CLI.md** : Documentation en ligne de commande.

## Prérequis

1. Clonez le dépôt :
   ```bash
   git clone https://votre-url-depot.git
   ```

2. crée vôtre environnement et activez le :
```bash
python3 -m venv venv
. venv/bin/activate
```
3. Installation des dépendances : 
    ```bash
   pip install -r requirements.txt
   ```

## Lancement des Scripts

1. **Accédez au répertoire contenant les scripts** que vous souhaitez exécuter.
    ```bash
    cd scripts/
     ```


2. **Lancez le script via la ligne de commande** en suivant les étapes et les exemples spécifiés dans le fichier [CLI.Md](CLI.Md).
