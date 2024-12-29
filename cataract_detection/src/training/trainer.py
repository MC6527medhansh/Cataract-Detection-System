import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.preprocessing.data_loader import get_dataloaders
from src.model.model import CataractCNN
from config.config import DEVICE, EPOCHS, LEARNING_RATE, MODELS_DIR
from tqdm import tqdm


def train():
    print("Starting training...")  # Debug print

    train_loader, val_loader, _ = get_dataloaders()
    print("Data loaders initialized.")  # Debug print

    model = CataractCNN(num_classes=5).to(DEVICE)
    print("Model initialized.")  # Debug print

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print("Criterion and optimizer initialized.")  # Debug print

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}...")  # Debug print
        model.train()
        train_loss = 0

        for images, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            try:
                print(f"Loaded a batch: images {images.shape}, labels {labels.shape}")  # Batch loaded

                images, labels = images.to(DEVICE), labels.to(DEVICE)
                print("Moved batch to device.")  # Debug print for device allocation

                optimizer.zero_grad()
                print("Optimizer gradients cleared.")  # Debug print for optimizer step

                outputs = model(images)
                print(f"Model outputs shape: {outputs.shape}")  # Debug print for model output

                loss = criterion(outputs, labels)
                print(f"Loss computed: {loss.item()}")  # Debug print for loss computation

                loss.backward()
                print("Loss backpropagated.")  # Debug print for backward pass

                optimizer.step()
                print("Optimizer step completed.")  # Debug print for optimizer step

                train_loss += loss.item()
                
            except Exception as e:
                print(f"Error processing batch: {e}")
                continue

        print(f"Epoch {epoch+1} training loss: {train_loss/len(train_loader)}")  # Debug print

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}"):
                print(f"Validation batch: images {images.shape}, labels {labels.shape}")
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        print(f"Epoch {epoch+1} validation loss: {val_loss/len(val_loader)}")  # Debug print

    print("Training completed.")  # Debug print

    # Save the model
    torch.save(model.state_dict(), f"{MODELS_DIR}/cataract_model.pth")
    print("Model saved.")


if __name__ == "__main__":
    print("Running training script...")

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Seed set to: {SEED}")

    train()
