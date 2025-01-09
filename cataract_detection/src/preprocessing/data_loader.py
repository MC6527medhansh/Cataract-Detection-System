import os
import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, WeightedRandomSampler


TRAIN_DIR = "cataract_detection/dataset/eyes.v1i.folder/train"
VAL_DIR = "cataract_detection/dataset/eyes.v1i.folder/valid"
TEST_DIR = "cataract_detection/dataset/eyes.v1i.folder/test"


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def get_dataloaders(batch_size=32):
    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_test_transform)
    test_dataset = datasets.ImageFolder(TEST_DIR, transform=val_test_transform)

    # WeightedRandomSampler for class imbalance
    class_counts = [len(os.listdir(os.path.join(TRAIN_DIR, cls))) for cls in train_dataset.classes]
    class_weights = [1.0 / count for count in class_counts]
    sample_weights = [class_weights[label] for label in train_dataset.targets]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders()
    print("Dataloaders created successfully.")
