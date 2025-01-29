import os
import random
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import medmnist
from medmnist import INFO
from torch.utils.data import DataLoader
import argparse

def preview_data(data_flag, num_per_class=10):
    """
    Preview MedMNIST training data.
    
    Args:
        data_flag (str): The dataset name from MedMNIST (e.g., 'pathmnist', 'octmnist').
        num_per_class (int): Number of images per class to display in the class-wise plot.
    """
    print(f"{data_flag} is going to be loaded.")
    preview_plot_folder = os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)), "Preview"))
    os.makedirs(preview_plot_folder, exist_ok = True)
    # Load dataset info
    info = INFO[data_flag]
    is_multi_label = info["task"] == "multi-label, binary-class"
    print(f"The data is multi-label: {is_multi_label}")
    DataClass = getattr(medmnist, info["python_class"])
    train_dataset = DataClass(split="train", transform=transforms.ToTensor(), download=True)
    print("Train data downloaded")
    # Load data loader
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
    images, labels = next(iter(train_loader))
    labels = labels.squeeze()

    # Dataset metadata
    classes = info["label"]  # Class names
    num_classes = len(classes)
    image_size = images.shape[-1]
    num_images = len(train_dataset)

    # First Plot: Random 10x10 Thumbnail
    def plot_random_thumbnails(images, labels):
        """
        Plot a random 10x10 grid of thumbnails from the dataset.
        """
        plt.figure(figsize=(15, 15))
        indices = random.sample(range(len(images)), 100)  # Randomly sample 100 images
        sampled_images = images[indices]
        sampled_labels = labels[indices]

        for i in range(100):
            # Convert image to (H, W, 3) format for RGB
            image = sampled_images[i].permute(1, 2, 0).numpy()
            if is_multi_label:
                active_labels = torch.nonzero(sampled_labels[i]).squeeze().tolist()  # Get indices of active labels

                active_labels = [active_labels] if isinstance(active_labels, int) else active_labels
                if active_labels == []:
                    label_text = "Normal"
                else:
                    label_text = ", ".join([classes[str(label)] for label in active_labels])  # Map to class names
                plt.subplot(10, 10, i + 1)
                plt.imshow(image)  # Automatically handles RGB
                plt.axis("off")
                plt.title(label_text, fontsize=8)
            else:
                plt.subplot(10, 10, i + 1)
                plt.imshow(image)  # Automatically handles RGB
                plt.axis("off")
                # plt.title(f"{sampled_labels[i].item()}", fontsize=8)
                plt.title(f"{classes[str(sampled_labels.squeeze().tolist()[i])]}", fontsize=8)

        plt.suptitle(f"Dataset: {data_flag} | Total Images: {num_images} | Image Size: {image_size}x{image_size}", 
                    fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(preview_plot_folder, f"{data_flag}_random_thumbnail.png"))
        plt.show()


    def plot_class_wise_thumbnails(images, labels, num_per_class):
        """
        Plot class-wise thumbnails, showing num_per_class images for each class.
        """
        plt.figure(figsize=(15, 2 * num_classes))
        if is_multi_label:
            class_indices = (labels[:, :] == 0).nonzero(as_tuple=True)[0]  # Find images with this label
            sampled_indices = random.sample(class_indices.tolist(), min(num_per_class, len(class_indices)))
            for i, idx in enumerate(sampled_indices):
                # Convert image to (H, W, 3) format for RGB
                image = images[idx].permute(1, 2, 0).numpy()
                plt.subplot(num_classes, num_per_class, i + 1)
                plt.imshow(image)
                plt.axis("off")

                # Add the class name on the left side of the row
                if i == 0:
                    plt.title("Normal", fontsize=10)

        for c in range(num_classes):
            if is_multi_label:
                class_indices = (labels[:, c] == 1).nonzero(as_tuple=True)[0]  # Find images with this label
            else:
                class_indices = (labels == c).nonzero(as_tuple=True)[0]
            sampled_indices = random.sample(class_indices.tolist(), min(num_per_class, len(class_indices)))

            for i, idx in enumerate(sampled_indices):
                # Convert image to (H, W, 3) format for RGB
                image = images[idx].permute(1, 2, 0).numpy()
                if is_multi_label:
                    plt.subplot(num_classes + 1, num_per_class, (c + 1) * num_per_class + i + 1)
                else:
                    plt.subplot(num_classes + 1, num_per_class, c * num_per_class + i + 1)
                plt.imshow(image)
                plt.axis("off")

                # Add the class name on the left side of the row
                if i == 0:
                    plt.title(classes[str(c)], fontsize=10)

        plt.suptitle(f"Class-Wise Thumbnail Plot ({num_per_class} Images per Class)", fontsize=16)
        # plt.tight_layout(rect=[0.05, 0, 1, 0.95])  # Add padding to the left
        plt.savefig(os.path.join(preview_plot_folder, f"{data_flag}_classwise_thumbnail.png"))
        plt.show()

    # Class counts
    def count_images_per_class(labels, num_classes):
        print(f"Total number of images is: {images.shape[0]}")
        if is_multi_label:
            counts = labels.sum(dim=0).int()  # Sum along samples to count per class
        else:
            counts = torch.bincount(labels, minlength=num_classes)
        for i, count in enumerate(counts):
            class_name = f"Class {i}" if classes is None else classes.get(str(i), f"Class {i}")
            print(f"{class_name}: {count.item()} images")
    
    # Generate the plots
    print("Generating random 10x10 thumbnail plot...")
    plot_random_thumbnails(images, labels)

    print("Generating class-wise thumbnail plot...")
    plot_class_wise_thumbnails(images, labels, num_per_class)

    print("Counting images per class...")
    count_images_per_class(labels, num_classes)


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", type=str, required=True, help="Dataset to load: 'pathmnist(default)', 'chestmnist', 'dermamnist', 'octmnist', 'pneumoniamnist', 'retinamnist', \
                        'breastmnist', 'bloodmnist', 'tissuemnist', 'organamnist', 'organcmnist', 'organsmnist'")
    parser.add_argument("-n", type=int, default=10, help="Number of images per class to preview")
    args = parser.parse_args()
    preview_data(data_flag=args.d, num_per_class=args.n)
