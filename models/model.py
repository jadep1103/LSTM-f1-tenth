import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size=7, hidden_size=128, num_classes=5, num_layers=2):
        super(LSTMModel, self).__init__()
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # Passer les données dans le LSTM
        out, _ = self.lstm(x)
        
        # Prendre la dernière sortie de la séquence
        out = out[:, -1, :]  # Prendre uniquement la dernière sortie (mouvement final)
        
        # Passer la sortie du LSTM dans une couche fully connected linéaire pour la classification
        out = self.fc(out)
        
        return out
