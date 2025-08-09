# train.py
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader, random_split
from models.model import LSTMModel  # Importer le modèle
from dataset.imudataset import IMUDataset  # Importer ton dataset
import matplotlib.pyplot as plt

# Charger les données
dataset = IMUDataset("/Users/jadepillercammal/f1tenth/F1-tenth/lstm/rawdata/imu_data.csv", window_size=50)


# Calcul des poids des classes (inverse de fréquence)
labels = np.array([sample[1].item() for sample in dataset])  # Récupérer tous les labels
print("Nombre de NaN:", np.isnan(labels).sum())
print("Labels min/max:", labels.min(), labels.max())
class_counts = np.bincount(labels)  # Nombre d'occurrences par classe
class_counts[class_counts == 0] = 1 
class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)  # Inverse des fréquences
class_weights /= class_weights.sum()  # Normalisation


# Séparation des données : 70% pour l'entraînement et 30% pour le test
train_size = int(0.7 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# DataLoader pour l'entraînement et le test
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialiser le modèle LSTM
model = LSTMModel(input_size=7, hidden_size=128, num_classes=5)

# Définir la loss et l'optimiseur
criterion = nn.CrossEntropyLoss(weight=class_weights)  # Pour classification multi-classe
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Boucle d'entraînement
num_epochs = 2
for epoch in range(num_epochs):
    model.train()
    
    for batch_idx, (sequences, labels) in enumerate(train_dataloader):
        optimizer.zero_grad()  # Remise à zéro des gradients
        
        # Forward pass
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Optimisation
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx}/{len(train_dataloader)}], Loss: {loss.item():.4f}")

# Sauvegarder le modèle après l'entraînement
# Générer un nom de fichier basé sur la date et l'heure
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
model_filename = f"lstm_model_{timestamp}.pth"

# Sauvegarder le modèle
torch.save(model.state_dict(), model_filename)
print(f"Modèle sauvegardé sous : {model_filename}")

# Évaluation sur un jeu de test
model.eval()  # Mettre le modèle en mode évaluation
with torch.no_grad():
    correct = 0
    total = 0
    for sequences, labels in test_dataloader:
        outputs = model(sequences)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy}%")
