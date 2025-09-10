import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse

def analyse_incertitude(dataframe_path, seuil_min, seuil_max, seuil_pas, resultat_path):
    """
    Analyse les données du dataframe généré par l'inférence.
    permet la visualisation de la répartition de l'incertitude.
    Calcule les proportions de "True" dans les "False" et vice versa pour chaque seuil 
    d'incertitude donné.
    Sauvegarde des résultats dans un fichier CSV et affichage des graphiques pour l'analyse de l'inférence.

    Arguments :
    dataframe_path -- chemin vers le fichier CSV contenant les données à analyser
    seuil_min -- seuil minimum d'incertitude pour l'analyse
    seuil_max -- seuil maximum d'incertitude pour l'analyse
    seuil_pas -- pas de variation des seuils
    resultat_path -- chemin où sauvegarder les résultats de l'analyse
    """
    
    df = pd.read_csv(dataframe_path, delimiter=';')

    correct_true = df[df['correct_std'] == True]
    correct_false = df[df['correct_std'] == False]

    uncertainty_entropy_true = correct_true['uncertainty_entropy']
    uncertainty_entropy_false = correct_false['uncertainty_entropy']

    plt.figure(figsize=(10, 6))
    sns.histplot(uncertainty_entropy_true, kde=True, color='blue', label='Correct STD == True', stat='density', bins=30)
    sns.histplot(uncertainty_entropy_false, kde=True, color='red', label='Correct STD == False', stat='density', bins=30)

    plt.title('Distribution de l\'incertitude (uncertainty_entropy)', fontsize=16)
    plt.xlabel('Incertitude Entropie', fontsize=12)
    plt.ylabel('Densité', fontsize=12)
    plt.legend()
    plt.savefig(resultat_path + '/distribution_entropie.png')
    plt.show()


    seuils = np.arange(seuil_min, seuil_max, seuil_pas)

    proportions_true_in_false = []
    for seuil in seuils:
        correct_false = correct_false.copy()
        correct_true = correct_true.copy()

        correct_false['uncertainty_entropy'] = pd.to_numeric(correct_false['uncertainty_entropy'], errors='coerce')
        correct_true['uncertainty_entropy'] = pd.to_numeric(correct_true['uncertainty_entropy'], errors='coerce')


        correct_false = correct_false.dropna(subset=['uncertainty_entropy'])
        correct_true = correct_true.dropna(subset=['uncertainty_entropy'])

        false_with_uncertainty_below_threshold = correct_false[correct_false['uncertainty_entropy'] < seuil]
        true_in_false = correct_true[correct_true['uncertainty_entropy'] < seuil]

        proportion_true_in_false = len(true_in_false) / len(false_with_uncertainty_below_threshold) if len(false_with_uncertainty_below_threshold) > 0 else 0
        proportions_true_in_false.append(proportion_true_in_false)

    resultats = pd.DataFrame({
        'Seuil': seuils,
        'Proportion True dans False': proportions_true_in_false
    })

    proportions_false_in_true = []
    for seuil in seuils:
        true_with_uncertainty_below_threshold = correct_true[correct_true['uncertainty_entropy'] < seuil]
        false_in_true = correct_false[correct_false['uncertainty_entropy'] < seuil]

        proportion_false_in_true = len(false_in_true) / len(true_with_uncertainty_below_threshold) if len(true_with_uncertainty_below_threshold) > 0 else 0
        proportions_false_in_true.append(proportion_false_in_true)

    resultats['Proportion False dans True'] = proportions_false_in_true

    print(resultats)

    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Seuil', y='Proportion True dans False', data=resultats, label='Proportion True dans False', color='blue')
    sns.lineplot(x='Seuil', y='Proportion False dans True', data=resultats, label='Proportion False dans True', color='red')

    plt.title('Proportions de True dans False et False dans True selon le seuil d\'incertitude', fontsize=16)
    plt.xlabel('Seuil d\'incertitude', fontsize=12)
    plt.ylabel('Proportion', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.savefig(resultat_path + '/True_in_False.png')
    plt.show()

    resultats.to_csv(resultat_path +'/analyse_results.csv', index=False, sep=';')

def main():

    parser = argparse.ArgumentParser(description="Analyse de l'incertitude et des proportions pour les données.")
    parser.add_argument('--dataframe_path', type=str, help='Chemin vers le fichier CSV contenant les données.')
    parser.add_argument('--seuil_min', type=float, help='Seuil minimum pour l\'analyse.')
    parser.add_argument('--seuil_max', type=float, help='Seuil maximum pour l\'analyse.')
    parser.add_argument('--seuil_pas', type=float, help='Pas pour les seuils (ex: 0.001).')
    parser.add_argument('--resultat_path', type=str, help='Chemin pour sauvegarder le fichier CSV des résultats.')


    args = parser.parse_args()

    analyse_incertitude(args.dataframe_path, args.seuil_min, args.seuil_max, args.seuil_pas, args.resultat_path)

if __name__ == "__main__":
    main()
