from PIL import Image
from pathlib import Path
import os
import shutil
import random
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def split_dataset_into_train_test(dataset_path, train_ratio=0.8):
    """
    Splits a dataset into training and testing directories.

    Args:
        dataset_path (str): Path to the dataset directory.
        train_ratio (float): Ratio of data to use for training (default is 0.8).
    """
    data_dir = Path(dataset_path)
    train_dir = data_dir.parent / "train"
    test_dir = data_dir.parent / "test"
    train_dir_str = str(train_dir)
    test_dir_str = str(test_dir)
    # Remove existing directories if they exist and create new ones
    for directory in [train_dir, test_dir]:
        if directory.exists():
            print("✅ Dataset already splitted into train/test!")
            return train_dir_str,test_dir_str
        directory.mkdir(parents=True, exist_ok=True)

    # Create train and test directories for each class
    for class_name in os.listdir(data_dir):
        class_path = data_dir / class_name
        if os.path.isdir(class_path):  # Check if it's a directory
            (train_dir / class_name).mkdir(parents=True, exist_ok=True)
            (test_dir / class_name).mkdir(parents=True, exist_ok=True)

            # List and shuffle images
            images = list(class_path.glob("*"))
            random.shuffle(images)

            # Split into train and test sets
            split_idx = int(train_ratio * len(images))
            train_images = images[:split_idx]
            test_images = images[split_idx:]

            # Move files to their respective directories
            for img_path in train_images:
                shutil.move(str(img_path), str(train_dir / class_name / img_path.name))

            for img_path in test_images:
                shutil.move(str(img_path), str(test_dir / class_name / img_path.name))

    print("✅ Dataset successfully split into train/test!")

    
    return train_dir_str,test_dir_str

def create_data_loaders(train_dir, test_dir, batch_size=32, image_size=(224, 224)):
    """
    Creates PyTorch DataLoader objects for training and testing datasets.

    Args:
        train_dir (str): Path to the training dataset directory.
        test_dir (str): Path to the testing dataset directory.
        batch_size (int): Batch size for DataLoader (default is 32).
        image_size (tuple): Size to resize images (default is (224, 224)).

    Returns:
        train_loader (DataLoader): DataLoader for training data.
        test_loader (DataLoader): DataLoader for testing data.
        classes (list): List of class names.
    """
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Load datasets using ImageFolder
    train_data = datasets.ImageFolder(root=train_dir, transform=transform)
    test_data = datasets.ImageFolder(root=test_dir, transform=transform)

    # Create DataLoader objects
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Print dataset information
    print("✅ Data loaders created successfully! Train/Test DataLoaders and Class Names Returned..")
    print("Classes:", train_data.classes)
    print("Total Training Data:", len(train_data))
    print("Total Testing Data:", len(test_data))
    return train_loader, test_loader, train_data.classes