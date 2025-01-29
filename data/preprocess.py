import medmnist
from medmnist import INFO
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import torch
from torchvision.transforms import Lambda

def get_train_data_loaders(data_flag='pathmnist', batch_size=32, val_split=0.2, seed=42):
    """
    Get DataLoaders for training and validation datasets. Supports both single-label and multi-label datasets.

    Args:
        data_flag (str): MedMNIST dataset flag (e.g., 'pathmnist', 'chestmnist').
        batch_size (int): Batch size for DataLoader.
        val_split (float): Fraction of the training data to use for validation.
        seed (int): Random seed for reproducibility.
    
    Returns:
        train_loader, val_loader: DataLoader for training and validation datasets.
    """
    # Get dataset info
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])
    is_multi_label = info['task'] == "multi-label, binary-class"

    # Define transforms
    if info["n_channels"] == 1:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            Lambda(lambda x: x.repeat(3, 1, 1)),  # Convert 1-channel to 3-channel
            transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
        ])
    elif info["n_channels"] == 3:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
        ])
    # Load full training dataset
    train_dataset = DataClass(split='train', transform=train_transform, download=True)
    val_dataset = DataClass(split='val', transform=train_transform, download=True)
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=(multi_label_collate_fn if is_multi_label else None)
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=(multi_label_collate_fn if is_multi_label else None)
    )
    
    return train_loader, val_loader


def get_test_data_loader(data_flag='pathmnist', batch_size=32):
    """
    Get DataLoader for the test dataset. Supports both single-label and multi-label datasets.

    Args:
        data_flag (str): MedMNIST dataset flag (e.g., 'pathmnist', 'chestmnist').
        batch_size (int): Batch size for DataLoader.

    Returns:
        test_loader: DataLoader for the test dataset.
    """
    # Get dataset info
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])
    is_multi_label = info['task'] == "multi-label, binary-class"

    # Define transforms
    if info["n_channels"] == 1:
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            Lambda(lambda x: x.repeat(3, 1, 1)),  # Convert 1-channel to 3-channel
            transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
        ])
    elif info["n_channels"] == 3:
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
        ])
    
    # Load test dataset
    test_dataset = DataClass(split='test', transform=test_transform, download=True)
    
    # Create DataLoader
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=(multi_label_collate_fn if is_multi_label else None)
    )
    
    return test_loader


def multi_label_collate_fn(batch):
    """
    Collate function for multi-label datasets. Converts labels to appropriate PyTorch tensors.

    Args:
        batch (list): List of (image, label) tuples.
    
    Returns:
        images: Tensor of images.
        labels: Tensor of multi-label targets.
    """
    images, labels = zip(*batch)
    images = torch.stack(images, dim=0)  # Stack images into a batch

    # Convert labels to tensors and stack them
    labels = torch.stack([torch.tensor(label, dtype=torch.float32) for label in labels], dim=0)
    return images, labels