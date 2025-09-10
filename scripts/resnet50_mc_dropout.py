# resnet50_mc_dropout.py

import torch
import torch.nn as nn
from torchvision.models import resnet50



class ResNet50_MCDropout(nn.Module):
    def __init__(self, base_model, num_classes, train_from='layer3', dropout_rate=0.3):
        super().__init__()

        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool

        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        self.avgpool = base_model.avgpool

        self.train_from = train_from
        self.dropout_rate = dropout_rate

        # Dropout après les blocs résiduels
        self.do_after_1 = nn.Dropout(dropout_rate) if train_from in ['layer1'] else nn.Identity()
        self.do_after_2 = nn.Dropout(dropout_rate) if train_from in ['layer1', 'layer2'] else nn.Identity()
        self.do_after_3 = nn.Dropout(dropout_rate) if train_from in ['layer1', 'layer2', 'layer3'] else nn.Identity()
        self.do_after_4 = nn.Dropout(dropout_rate) if train_from in ['layer1', 'layer2', 'layer3', 'layer4'] else nn.Identity()

        # Tête de classification
        self.fc = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

        self._freeze_layers(base_model)

    def _freeze_layers(self, base_model):
        """ Gèle toutes les couches sauf à partir de train_from """
        freeze_before = {
            'layer1': [],
            'layer2': list(base_model.layer1.parameters()),
            'layer3': list(base_model.layer1.parameters()) + list(base_model.layer2.parameters()),
            'layer4': list(base_model.layer1.parameters()) + list(base_model.layer2.parameters()) + list(base_model.layer3.parameters()),
        }
        for param in base_model.parameters():
            param.requires_grad = False
        for param in freeze_before.get(self.train_from, []):
            param.requires_grad = False
        for param in self.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.do_after_1(x)

        x = self.layer2(x)
        x = self.do_after_2(x)

        x = self.layer3(x)
        x = self.do_after_3(x)

        x = self.layer4(x)
        x = self.do_after_4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x