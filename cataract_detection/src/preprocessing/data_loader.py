import os
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split

# Dataset paths
TRAIN_DIR = "cataract_detection/Ocular Disease Classification.v11i.folder/train"
TEST_DIR = "cataract_detection/Ocular Disease Classification.v11i.folder/test"

# Image transformations
train_transform = transforms.Compose([
    transforms.Resize((640, 640)),  # Match dataset preprocessing
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_test_transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Class names
CLASS_NAMES = [
    "age related macular degeneration",
    "cataract",
    "diabetes",
    "diabetes glaucoma",
    "diabetes hypertension",
    "glaucoma",
    "hypertension",
    "normal",
]


def get_dataloaders(batch_size=64, val_split=0.2):
    # Load the training dataset
    full_train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)

    # Split into training and validation sets
    train_size = int((1 - val_split) * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    # Test dataset
    test_dataset = datasets.ImageFolder(TEST_DIR, transform=val_test_transform)

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
