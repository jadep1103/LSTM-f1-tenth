import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class IMUDataset(Dataset):
    def __init__(self, file_path, window_size=50):
        df = pd.read_csv(file_path)

        # Lister les colonnes
        features = ["ax", "ay", "az", "qx", "qy", "qz", "qw"]
        labels = df["label"].values
        #Remplacer les labels 1 et 2 par 0
        labels[labels == 1] = 0
        labels[labels == 2] = 0

        
        # Normalisation -> TO CHECK
        df[features] = (df[features] - df[features].mean()) / df[features].std()
        
        self.data = df[features].values
        self.labels = labels
        self.window_size = window_size
        self.num_samples = len(df) - window_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Retourne une séquence de `window_size` timestamps et le label associé.
        """
        sequence = self.data[idx:idx + self.window_size]
        label = self.labels[idx + self.window_size - 1]  # Label du dernier timestamp de la fenêtre
        return torch.tensor(sequence, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

if __name__ == "__main__":

    dataset = IMUDataset("/Users/jadepillercammal/f1tenth/F1-tenth/lstm/rawdata/imu_data.csv", window_size=50)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # # Affichage dimensions d'un exemple
    # sample, label = dataset[0]
    # print("Sample shape:", sample.shape)  # (50, 7) -> taille de la fenêtre x nombre de features
    # print("Label:", label)

    # Affichage d'un exemple
    samples, labels = next(iter(dataloader))
    print(f"Sample batch shape: {samples.shape}")  
    print(f"Labels: {labels}") 

    example_idx = 1 
    sample = samples[example_idx].numpy()

    # Tracer les accéléromètres (ax, ay, az)
    plt.figure(figsize=(10, 4))
    plt.plot(sample[:, 0], label="ax", color="r")
    plt.plot(sample[:, 1], label="ay", color="g")
    plt.plot(sample[:, 2], label="az", color="b")
    plt.legend()
    plt.title(f"Accéléromètre - Label: {labels[example_idx].item()}")
    plt.show()