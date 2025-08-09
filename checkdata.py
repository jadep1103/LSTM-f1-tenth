# import pandas as pd

# # Charger les données brutes (CSV)
# data = pd.read_csv("/Users/jadepillercammal/f1tenth/F1-tenth/lstm/rawdata/imu_data.csv")



# # Afficher les premières lignes du fichier pour avoir un aperçu
# print("Aperçu des données brutes :")
# print(data.head())

# # Vérifier la répartition des labels
# print("\nRépartition des labels dans les données :")
# print(data['label'].value_counts())

# # Mélanger les données de manière aléatoire
# data_shuffled = data.sample(frac=1, random_state=42).reset_index(drop=True)

# # Afficher un échantillon de données mélangées
# print("\nAperçu des données mélangées :")
# print(data_shuffled.head())

# # Vérifier la répartition des labels après mélange
# print("\nRépartition des labels dans les données mélangées :")
# print(data_shuffled['label'].value_counts())

import pandas as pd
import numpy as np

# Charger les données brutes (CSV)
file_path = "/Users/jadepillercammal/f1tenth/F1-tenth/lstm/rawdata/imu_data.csv"
data = pd.read_csv(file_path)

# Afficher les premières lignes du fichier pour avoir un aperçu
print("Aperçu des données brutes :")
print(data.head())

# Remplacement des classes 1 et 2 par 0
data["label"] = data["label"].replace({1: 0, 2: 0})

# Vérifier la répartition des labels après modification
print("\nRépartition des labels après regroupement :")
print(data["label"].value_counts())

# Vérification du déséquilibre des classes
unique, counts = np.unique(data["label"], return_counts=True)
class_distribution = dict(zip(unique, counts))

print("\nVérification du déséquilibre des classes :")
for label, count in class_distribution.items():
    print(f"Classe {label} : {count} échantillons")

# Vérifier si le dataset est fortement déséquilibré
max_class = max(counts)
min_class = min(counts)

if max_class / min_class > 3:  # Seuil arbitraire pour un fort déséquilibre
    print("Alerte : Déséquilibre significatif détecté !")
    print("Il pourrait être utile d'appliquer une pondération dans la loss ou un échantillonnage équilibré.")
else:
    print("Les classes semblent bien équilibrées.")

# Mélanger les données de manière aléatoire
data_shuffled = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Afficher un échantillon de données mélangées
print("\nAperçu des données mélangées :")
print(data_shuffled.head())

# Vérifier la répartition des labels après mélange
print("\nRépartition des labels dans les données mélangées :")
print(data_shuffled["label"].value_counts())
