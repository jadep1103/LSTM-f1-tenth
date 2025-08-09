import torch
from torch.utils.data import DataLoader
from models.model import LSTMModel
from dataset.imudataset import IMUDataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

def load_model(model_path="lstm_model.pth"):
    model = LSTMModel(input_size=7, hidden_size=128, num_classes=3)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Passer en mode évaluation
    return model

def load_data(data_path="/Users/jadepillercammal/f1tenth/F1-tenth/lstm/rawdata/imu_data.csv", window_size=50):
    dataset = IMUDataset(data_path, window_size)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    return dataloader

def get_predictions(model, dataloader):
    predictions = []
    true_labels = []
    examples = []  # Pour stocker quelques exemples
    
    with torch.no_grad():
        for i, (sequences, labels) in enumerate(dataloader):
            outputs = model(sequences)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.tolist())
            true_labels.extend(labels.tolist())
            
            # Sauvegarder quelques exemples pour affichage
            if i * 32 < 15:  # Limite à 15 exemples pour l'affichage
                for seq, label, pred in zip(sequences, labels, predicted):
                    examples.append((seq.tolist(), label.item(), pred.item()))  # Ajouter l'exemple sous forme de liste
    
    return true_labels, predictions, examples

def display_examples(examples):
    # Affichage de quelques exemples avec leurs prédictions
    for i, (data, true_label, pred_label) in enumerate(examples):
        print(f"Exemple {i+1}:")
        print(f"    Données : {data}")  # Afficher les données (en liste)
        print(f"    Vrai label : {true_label}")
        print(f"    Prédiction : {pred_label}")
        print()

def calculate_metrics(true_labels, predictions):
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='weighted')
    conf_matrix = confusion_matrix(true_labels, predictions)
    
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"F1-Score: {f1:.2f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    
    return conf_matrix

def plot_confusion_matrix(conf_matrix):
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Matrice de Confusion')
    plt.colorbar()
    tick_marks = range(len(conf_matrix))
    plt.xticks(tick_marks, range(len(conf_matrix)))
    plt.yticks(tick_marks, range(len(conf_matrix)))
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

if __name__ == "__main__":
    model = load_model()
    dataloader = load_data()
    true_labels, predictions, examples = get_predictions(model, dataloader)
    display_examples(examples)  # Afficher les exemples
    conf_matrix = calculate_metrics(true_labels, predictions)
    plot_confusion_matrix(conf_matrix)
