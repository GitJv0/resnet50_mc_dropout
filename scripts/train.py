# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score
from tqdm import tqdm
import pandas as pd
import argparse
import time
import os
from datetime import datetime
from torchvision.models import resnet50
from resnet50_mc_dropout import ResNet50_MCDropout

from utils.dataloader_with_aug import AlbumentationDataModule

def check_gpu():
    if not torch.cuda.is_available():
        print("❌ Aucun GPU détecté. Le script nécessite un GPU.")
        exit(1)
    print(f"✅ GPU utilisé : {torch.cuda.get_device_name(0)}")


def build_model(num_classes, resume_path=None, device='cuda', train_from='layer3', dropout_rate=0.3, num_classes_base_model= 3):
    base = resnet50(weigths=None)
    model = ResNet50_MCDropout(base_model=base, num_classes=num_classes, train_from=train_from, dropout_rate=dropout_rate)

    if resume_path and os.path.exists(resume_path):
        print(f"🔁 Chargement modèle depuis : {resume_path}")

        # Tête de classification
        model.fc = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes_base_model)
        )

        model.load_state_dict(torch.load(resume_path, map_location=device))

        model.fc = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

        
    else:
        print(f"🧠 Modèle ResNet50 + Dropout inséré à partir de : {train_from}")

    return model

def train_model(model, dataloader, device, optimizer, loss_fn, scheduler, metrics, num_classes, epochs, output_path):
    best_loss = float('inf')
    results = []

    for epoch in range(epochs):
        print(f"\n🔄 Epoch {epoch+1}/{epochs}")
        start_time = time.time()
        model.train()
        running_loss = 0.0
        for m in metrics.values():
            m.reset()

        loop = tqdm(dataloader['train'], desc="Training", leave=False)
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            for metric in metrics.values():
                metric.update(preds, labels)

        train_loss = running_loss / len(dataloader['train'])
        train_metrics = {name: metric.compute().item() for name, metric in metrics.items()}

        # Validation
        model.eval()
        val_loss = 0.0
        for m in metrics.values():
            m.reset()

        with torch.no_grad():
            for images, labels in dataloader['val']:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                for metric in metrics.values():
                    metric.update(preds, labels)

        val_loss /= len(dataloader['val'])
        val_metrics = {name: metric.compute().item() for name, metric in metrics.items()}
        scheduler.step(val_loss)
        elapsed = time.time() - start_time

        print(
            f"✅ Epoch {epoch+1} | "
            f"Train Loss: {train_loss:.4f} | Acc: {train_metrics['accuracy']:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | "
            f"F1: {val_metrics['f1']:.4f} | Time: {elapsed:.1f}s"
        )

        results.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_accuracy': train_metrics['accuracy'],
            'train_precision': train_metrics['precision'],
            'train_recall': train_metrics['recall'],
            'train_f1': train_metrics['f1'],
            'val_loss': val_loss,
            'val_accuracy': val_metrics['accuracy'],
            'val_precision': val_metrics['precision'],
            'val_recall': val_metrics['recall'],
            'val_f1': val_metrics['f1'],
            'lr': optimizer.param_groups[0]['lr'],
            'time': elapsed
        })

        pd.DataFrame(results).to_csv(os.path.join(output_path, 'results.csv'), index=False)
        torch.save(model.state_dict(), os.path.join(output_path, 'last.pt'))
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(output_path, 'best.pt'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help="Chemin vers le dossier contenant train/val")
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--resume', type=str, default=None, help="Chemin vers un modèle .pt à recharger")
    parser.add_argument('--num-classes-base-model', type=int, default=3, help="Nombre de classe du model resume")
    parser.add_argument('--train-from', type=str, default='layer3', choices=['layer1', 'layer2', 'layer3', 'layer4'], help="À partir de quelle couche on applique le dropout et le fine-tuning")
    parser.add_argument('--dropout', type=float, default=0.3, help="Taux de Dropout appliqué à partir de la couche choisie")
    parser.add_argument('--no-aug', action='store_true', help="Désactive l'augmentation de données Albumentations")
    args = parser.parse_args()

    check_gpu()
    device = torch.device("cuda")

    
    # dataloaders, num_classes, _ = get_dataloaders(args.data, args.batch_size)
    data_module = AlbumentationDataModule(
    data_dir=args.data,
    batch_size=args.batch_size,
    use_aug=not args.no_aug
    )
    dataloaders, num_classes, _ = data_module.get_dataloaders()


    model = build_model(
        num_classes,
        resume_path=args.resume,
        device=device,
        train_from=args.train_from,
        dropout_rate=args.dropout,
        num_classes_base_model=args.num_classes_base_model
    ).to(device)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join("../runs", f"experiment_{timestamp}")
    os.makedirs(output_path, exist_ok=True)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5, verbose=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    metrics = {
        'precision': Precision(task='multiclass', num_classes=num_classes, average='macro').to(device),
        'recall': Recall(task='multiclass', num_classes=num_classes, average='macro').to(device),
        'f1': F1Score(task='multiclass', num_classes=num_classes, average='macro').to(device),
        'accuracy': Accuracy(task='multiclass', num_classes=num_classes, average='macro').to(device)
    }

    train_model(model, dataloaders, device, optimizer, loss_fn, scheduler, metrics, num_classes, args.epochs, output_path)