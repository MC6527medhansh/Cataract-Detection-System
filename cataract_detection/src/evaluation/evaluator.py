import torch
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.preprocessing.data_loader import get_dataloaders
from src.model.model import CataractCNN
from config.config import DEVICE, MODELS_DIR


def evaluate_model():
    model = CataractCNN(num_classes=4)  
    model.load_state_dict(torch.load(f"{MODELS_DIR}/cataract_model.pth"))
    model.to(DEVICE)
    model.eval()

    _, _, test_loader = get_dataloaders(batch_size=32)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=["Cataract", "Diabetic Retinopathy", "Glaucoma", "Normal"]))

    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Cataract", "Diabetic Retinopathy", "Glaucoma", "Normal"], yticklabels=["Cataract", "Diabetic Retinopathy", "Glaucoma", "Normal"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()
