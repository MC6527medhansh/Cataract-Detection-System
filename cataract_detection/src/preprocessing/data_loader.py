import os
import sys
import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.config import IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE

# Define dataset paths
TRAIN_DIR = "cataract_detection/dataset/augmented_resized_V2/train"
VAL_DIR = "cataract_detection/dataset/augmented_resized_V2/val"
TEST_DIR = "cataract_detection/dataset/augmented_resized_V2/test"

# Define transformations
train_transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_test_transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# Load datasets and create dataloaders
def get_dataloaders():
    print("Initializing data loaders...")  # Debug print

    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    print(f"Classes in training dataset: {train_dataset.classes}")  # Print classes

    val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_test_transform)
    print("Validation dataset loaded.")  # Debug print

    test_dataset = datasets.ImageFolder(TEST_DIR, transform=val_test_transform)
    print("Test dataset loaded.")  # Debug print

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("Data loaders created successfully!")  # Debug print
    return train_loader, val_loader, test_loader


# Test dataset loading
if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders()

    # Print some batch information
    for images, labels in train_loader:
        print(f"Image batch shape: {images.shape}")
        print(f"Label batch shape: {labels.shape}")
        break


# Test dataset paths
if __name__ == "__main__":
    print("Checking dataset paths:")
    for split, path in zip(["Train", "Validation", "Test"], [TRAIN_DIR, VAL_DIR, TEST_DIR]):
        if os.path.exists(path):
            print(f"{split} directory exists: {path}")
            print(f"Number of classes in {split}: {len(os.listdir(path))}")
        else:
            print(f"{split} directory NOT FOUND: {path}")
