import pandas as pd
import random
from PIL import Image
import matplotlib.pyplot as plt
import os
import argparse


def main(dataframe_path, seuil, resultat_path):
    
    df = pd.read_csv(dataframe_path, delimiter=';')

    df['uncertainty_entropy'] = pd.to_numeric(df['uncertainty_entropy'], errors='coerce')
    df = df.dropna(subset=['uncertainty_entropy'])

    correct_true = df[df['correct_std'] == True]
    correct_false = df[df['correct_std'] == False]

    above_threshold_true = correct_true[correct_true['uncertainty_entropy'] > seuil]
    below_threshold_true = correct_true[correct_true['uncertainty_entropy'] <= seuil]
    above_threshold_false = correct_false[correct_false['uncertainty_entropy'] > seuil]
    below_threshold_false = correct_false[correct_false['uncertainty_entropy'] <= seuil]

    percentage_above_threshold_true = (len(above_threshold_true) / len(correct_true)) * 100 if len(correct_true) > 0 else 0
    percentage_above_threshold_false = (len(above_threshold_false) / len(correct_false)) * 100 if len(correct_false) > 0 else 0

    print(f"Pourcentage des données correctes avec une incertitude > {seuil}: {percentage_above_threshold_true:.2f}%")
    print(f"Pourcentage des données incorrectes avec une incertitude > {seuil}: {percentage_above_threshold_false:.2f}%")

    cluster_summary = df.groupby('cluster').apply(lambda group: pd.Series({
        'correct_true_below_threshold': len(group[(group['correct_std'] == True) & (group['uncertainty_entropy'] <= seuil)]),
        'correct_true_above_threshold': len(group[(group['correct_std'] == True) & (group['uncertainty_entropy'] > seuil)]),
        'correct_false_below_threshold': len(group[(group['correct_std'] == False) & (group['uncertainty_entropy'] <= seuil)]),
        'correct_false_above_threshold': len(group[(group['correct_std'] == False) & (group['uncertainty_entropy'] > seuil)]),
        'total_correct_true': len(group[group['correct_std'] == True]),
        'total_correct_false': len(group[group['correct_std'] == False]),
    }))

    print(cluster_summary)

    os.makedirs(resultat_path, exist_ok=True)
    csv_path = os.path.join(resultat_path, "cluster_summary.csv")
    cluster_summary.to_csv(csv_path, index=True)
    print(f"Résumé sauvegardé dans {csv_path}")

    if len(above_threshold_true) >= 10:
        sample_images = random.sample(above_threshold_true['image_path'].tolist(), 10)
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.flatten()
        for idx, img_path in enumerate(sample_images):
            try:
                img = Image.open(img_path)
                axes[idx].imshow(img)
                axes[idx].axis('off')
                axes[idx].set_title(f"Image {idx + 1}")
            except Exception as e:
                print(f"Erreur ouverture image {img_path}: {e}")
        plt.tight_layout()
        plt.show()

    fig, ax = plt.subplots(figsize=(12, 7))
    cluster_summary[['correct_true_below_threshold', 'correct_true_above_threshold']].plot(
        kind='bar', stacked=True, ax=ax, position=1, color=['#67a9cf', '#1c5e7c'], width=0.4, label='Correct True')
    cluster_summary[['correct_false_below_threshold', 'correct_false_above_threshold']].plot(
        kind='bar', stacked=True, ax=ax, position=0, color=['#f4a582', '#d7301f'], width=0.4, label='Correct False')

    ax.set_title('Distribution des Images Correctes et Incorrectes par Cluster', fontsize=16)
    ax.set_xlabel('Cluster', fontsize=12)
    ax.set_ylabel('Nombre d\'Images', fontsize=12)
    ax.set_xticklabels(cluster_summary.index, rotation=45)
    ax.legend(title='Type de Prédiction', loc='upper right')

    plt.tight_layout()
    plot_path = os.path.join(resultat_path, "cluster_distribution.png")
    plt.savefig(plot_path)
    print(f"Graphique sauvegardé dans {plot_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse des clusters avec seuil d'entropie")
    parser.add_argument("--dataframe_path", type=str, required=True, help="Chemin du fichier CSV d'entrée")
    parser.add_argument("--seuil", type=float, required=True, help="Valeur du seuil d'entropie")
    parser.add_argument("--resultat_path", type=str, required=True, help="Dossier de sortie pour les résultats")

    args = parser.parse_args()
    main(args.dataframe_path, args.seuil, args.resultat_path)
