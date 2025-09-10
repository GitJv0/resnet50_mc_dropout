# Activer l'environnement
. venv/bin/activate

# Attention la structure du dossier data doit être train, val avec des dossiers qui contiennt le noms des classes et les images à l'intérieur des deux dossiers.

# train.py
python script.py \
    --data /chemin/vers/le/dossier \
    --epochs 10 \
    --batch-size 32 \
    --resume /chemin/vers/le/model.pt \
    --num-classes-base-model 3 \
    --train-from layer3 \
    --dropout 0.3 \
    --no-aug


python train.py \
    --data ../data/finetuning_resnet_train_val \
    --epochs 2 \
    --batch-size 32 \
    --resume ../models/ResNet50_Background_layer_3.pt \
    --num-classes-base-model 2 \
    --train-from layer3 \
    --dropout 0.3 \
    --no-aug

# inference 

python inference.py \
    --model ../runs/experiment_20250904_103923/best.pt \
    --data ../data/finetuning_resnet_train_val \
    --samples 30 \
    --train-from layer3 \
    --dropout 0.3 \
    --out eval_report.csv \
    --pred-mc-incert-seuil 0.5 \
    --pred-mc-entropy-seuil 0.5 \

# Analyse

python3 analyse.py \
    --dataframe_path ../runs/experiment_20250904_103923/eval_report.csv \
    --seuil_min 0.0 \
    --seuil_max 1.0 \
    --seuil_pas 0.001 \
    --resultat_path ../runs/experiment_20250904_103923

# Analyse clusters
python analyse_clusters.py \
  --dataframe_path ../runs/experiment_20250904_103923/eval_report.csv \
  --seuil 0.05 \
  --resultat_path ../runs/experiment_20250904_103923





