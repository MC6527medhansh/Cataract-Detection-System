import torch
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.preprocessing.data_loader import get_dataloaders
from src.model.model import CataractCNN
from config.config import DEVICE, EPOCHS, LEARNING_RATE, MODELS_DIR
from tqdm import tqdm


def evaluate_model():
    # Load the model
    model = CataractCNN(num_classes=5)
    model.load_state_dict(torch.load(f"{MODELS_DIR}/cataract_model.pth"))
    model.to(DEVICE)
    model.eval()

    # Load the test data
    _, _, test_loader = get_dataloaders()

    # Store all predictions and ground truths
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Processing batches"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Classification report
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=test_loader.dataset.classes))

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_loader.dataset.classes, yticklabels=test_loader.dataset.classes)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()


if __name__ == "__main__":
    evaluate_model()
