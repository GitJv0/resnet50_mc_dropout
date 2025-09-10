# dataloader_with_aug.py

import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset


class AlbumentationDataset(Dataset):
    def __init__(self, image_folder_dataset, transform=None):
        self.image_folder_dataset = image_folder_dataset
        self.transform = transform

    def __getitem__(self, idx):
        img_path, label = self.image_folder_dataset.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = np.array(image)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image, label

    def __len__(self):
        return len(self.image_folder_dataset)

import numpy as np

class AlbumentationDataModule:
    def __init__(self, data_dir, batch_size=32, use_aug=True):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.use_aug = use_aug

        self.train_transform = A.Compose([
            A.Resize(256, 256),                                # Standardise la taille en entrée
            A.RandomCrop(224, 224),                            # Focus sur différentes zones

            A.HorizontalFlip(p=0.5),                           # Miroir gauche-droite
            A.VerticalFlip(p=0.2),                             # Miroir haut-bas plus rare
            A.Rotate(limit=10, p=0.3),                         # Petites rotations, préserve la structure

            A.RandomBrightnessContrast(brightness_limit=0.1,
                                    contrast_limit=0.1, p=0.4),  # Légères variations lumière/contraste
            A.HueSaturationValue(hue_shift_limit=5,
                                sat_shift_limit=10,
                                val_shift_limit=5, p=0.3),        # Simule des différences de ton matière
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])


        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

    def get_dataloaders(self):
        train_folder = datasets.ImageFolder(os.path.join(self.data_dir, 'train'))
        val_folder = datasets.ImageFolder(os.path.join(self.data_dir, 'val'), transform=self.val_transform)

        if self.use_aug:
            train_dataset = AlbumentationDataset(train_folder, transform=self.train_transform)
        else:
            train_dataset = datasets.ImageFolder(os.path.join(self.data_dir, 'train'), transform=self.val_transform)

        val_dataset = val_folder

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

        return {"train": train_loader, "val": val_loader}, len(train_folder.classes), train_folder.classes


def get_train_transform():
    return AlbumentationDataModule(data_dir=".", use_aug=True).train_transform