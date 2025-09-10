import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from resnet50_mc_dropout import ResNet50_MCDropout
import pandas as pd
import argparse
import numpy as np
from tqdm import tqdm
import os
from scipy.stats import entropy

@torch.no_grad()
def predict_std(model, x, device):
    """
    Fonction pour effectuer une prédiction standard (sans dropout) sur un batch d'images.
    Elle renvoie les probabilités des classes pour chaque image du batch.

    Arguments :
    model -- modèle de réseau de neurones
    x -- batch d'images à prédire
    device -- périphérique (CPU ou GPU) où les calculs doivent être effectués

    Retour :
    Probabilités des classes pour chaque image dans le batch
    """
    model.eval()
    out = model(x.to(device))
    return F.softmax(out, dim=1).cpu().numpy()  # Retourner tous les résultats

def enable_dropout_only(model):
    """
    Active uniquement les couches de Dropout du modèle.

    Arguments :
    model -- modèle
    """
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()  # On garde Dropout activé
        elif isinstance(module, torch.nn.BatchNorm2d):
            module.eval()   # On force BatchNorm en mode eval

@torch.no_grad()
def predict_mc(model, x, device, n_samples=30):
    """
    Fonction pour effectuer une prédiction avec Dropout Monte Carlo sur un batch d'images.
    Elle génère plusieurs prédictions avec le Dropout activé, puis calcule la moyenne et l'écart-type des probabilités différentes classes.

    Arguments :
    model -- modèle avec Dropout activé
    x -- batch d'images à prédire
    device -- (CPU ou GPU) 
    n_samples -- nombre d'échantillons pour la simulation de Dropout (par défaut 30)

    Retour :
    Moyenne et écart-type des probabilités des classes pour chaque image du batch
    """
    enable_dropout_only(model)
    probs = []
    for _ in range(n_samples):
        out = model(x.to(device))
        probs.append(F.softmax(out, dim=1).cpu().numpy())
    probs = np.array(probs)
    mean = probs.mean(axis=0)  # Moyenne sur le batch
    std = probs.std(axis=0)    # Ecart-type sur le batch
    return mean, std

def load_model(path, num_classes, train_from='layer3', dropout=0.3, device='cuda'):
    """
    Charge un modèle ResNet50 avec Dropout Monte Carlo à partir d'un fichier de poids pré-entrainé.

    Arguments :
    path -- chemin vers le fichier contenant les poids du modèle
    num_classes -- nombre de classes pour la classification
    train_from -- couche de départ à partir de laquelle l'entraînement est effectué (par défaut 'layer3')
    dropout -- taux de Dropout à appliquer (par défaut 0.3)
    device --  (CPU ou GPU).

    Retour :
    Modèle chargé sur le device spécifié
    """
    base = resnet50(weights=None)
    model = ResNet50_MCDropout(base, num_classes, train_from, dropout)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    return model

def check_incertitude(class_names, pred_mc, pred_mc_incert, pred_mc_incert_seuil):
    """
    Vérifie si l'incertitude est inférieure ou supérieure à un seuil pour déterminer la classe prédite.
    
    Arguments :
    class_names -- liste des noms des classes
    pred_mc -- prédiction du modèle (indice de la classe)
    pred_mc_incert -- incertitude du modèle (écart-type de la prédiction)

    Retour :
    Classe prédite ou "incertitude" en fonction du seuil d'incertitude
    """
    if pred_mc_incert <=  pred_mc_incert_seuil:
        return class_names[pred_mc]
    else: 
        return 'incertitude'

def check_entropy(class_names, pred_mc, pred_mc_entropy, pred_mc_entropy_seuil):
    """
    Vérifie si l'entropie est inférieure ou supérieure à un seuil pour déterminer la classe prédite.

    Arguments :
    class_names -- liste des noms des classes
    pred_mc -- prédiction du modèle (indice de la classe)
    pred_mc_entropy -- entropie de la prédiction (indicateur d'incertitude)

    Retour :
    Classe prédite ou "incertitude" en fonction du seuil d'entropie
    """
    if pred_mc_entropy <= pred_mc_entropy_seuil:
        return class_names[pred_mc]
    else: 
        return 'incertitude'

def main():
    """
    Fonction principale pour exécuter l'évaluation du modèle avec Dropout Monte Carlo.
    Cette fonction charge les arguments de la ligne de commande, charge le modèle, 
    effectue les prédictions et les analyse, puis enregistre les résultats dans un fichier CSV.

    Arguments (récupérés via argparse) :
    --model -- chemin vers le fichier du modèle
    --data -- chemin vers le répertoire des données
    --samples -- nombre d'échantillons pour la simulation de Dropout (par défaut 30)
    --train-from -- couche de départ pour l'entraînement du modèle (par défaut 'layer3')
    --dropout -- taux de Dropout (par défaut 0.3)
    --out -- chemin pour sauvegarder le rapport des résultats (par défaut 'eval_report.csv')
    --pred-mc-incert-seuil -- seuil d'incertitude pour la décision de classe (par défaut 0.5)
    --pred-mc-entropy-seuil -- seuil d'entropie pour la décision de classe (par défaut 0.5)
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--samples', type=int, default=30)
    parser.add_argument('--train-from', default='layer3')
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--out', default='eval_report.csv')
    parser.add_argument('--pred-mc-incert-seuil', type=float, default=0.5, help='Seuil pour l\'incertitude (par défaut 0.5)')
    parser.add_argument('--pred-mc-entropy-seuil', type=float, default=0.5, help='Seuil pour l\'entropie (par défaut 0.5)')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(os.path.join(args.data, 'val'), transform=transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    class_names = dataset.classes

    model = load_model(args.model, len(class_names), args.train_from, args.dropout, device)

    results = []

    num_batches = len(dataset) // loader.batch_size

    for i, (x, y) in enumerate(tqdm(loader, total=num_batches)):
        true_classes = [class_names[label.item()] for label in y]
        true_idx = y.numpy()

        start_idx = i * loader.batch_size
        end_idx = min((i + 1) * loader.batch_size, len(dataset.imgs))  # Eviter les indices hors limites
        image_paths = [dataset.imgs[idx][0] for idx in range(start_idx, end_idx)]

        # Prédiction standard sur le batch
        probs_std = predict_std(model, x, device)
        preds_std = np.argmax(probs_std, axis=1)  # Prédictions pour tout le batch
        conf_std = probs_std[np.arange(len(preds_std)), preds_std]  # Confiance pour chaque prédiction
        std_correct = preds_std == true_idx  # Comparaison pour le batch

        # Prédiction MC Dropout sur le batch
        probs_mc, std_mc = predict_mc(model, x, device, args.samples)
        preds_mc = np.argmax(probs_mc, axis=1)  # Prédictions pour tout le batch
        conf_mc = probs_mc[np.arange(len(preds_mc)), preds_mc]  # Confiance pour chaque prédiction
        mc_correct = preds_mc == true_idx  # Comparaison pour le batch

        # Mesures d'incertitude
        uncertainty_std = std_mc[np.arange(len(preds_mc)), preds_mc]  # Incertitude par écart-type
        uncertainty_entropy = np.array([entropy(p) for p in probs_mc])  # Entropie pour chaque échantillon

        # Enregistrement des résultats
        for idx in range(len(x)):
            results.append({
                'image_index': start_idx + idx,  
                'image_path': image_paths[idx],  
                'true_class': true_classes[idx],
                'pred_std': class_names[preds_std[idx]],
                'correct_std': std_correct[idx],
                'conf_std': round(conf_std[idx], 4),
                'pred_mc_incertitude': check_incertitude(class_names, preds_mc[idx], uncertainty_std[idx], args.pred_mc_incert_seuil),
                'pred_mc_entropy': check_entropy(class_names, preds_mc[idx], uncertainty_entropy[idx], args.pred_mc_entropy_seuil),
                'correct_mc': mc_correct[idx],
                'conf_mc': round(conf_mc[idx], 4),
                'uncertainty_std': round(uncertainty_std[idx], 4),
                'uncertainty_entropy': round(uncertainty_entropy[idx], 4),
            })

    df = pd.DataFrame(results)
    df.to_csv(args.model.rsplit('/', 1)[0] + '/' + args.out, index=False, sep=';')
    print(f"\n✅ Rapport enregistré dans : {args.out}")

    wrong = df[df['correct_mc'] == False]
    right = df[df['correct_mc'] == True]
    print("\n📈 Analyse des incertitudes :")
    print(f" - Total images : {len(df)}")
    print(f" - Erreurs (MC) : {len(wrong)}")
    print(f" - Moy. incertitude_std (FAUX) : {wrong['uncertainty_std'].mean():.4f}")
    print(f" - Moy. incertitude_std (VRAI) : {right['uncertainty_std'].mean():.4f}")
    print(f" - Moy. entropie (FAUX) : {wrong['uncertainty_entropy'].mean():.4f}")
    print(f" - Moy. entropie (VRAI) : {right['uncertainty_entropy'].mean():.4f}")

if __name__ == "__main__":
    main()
