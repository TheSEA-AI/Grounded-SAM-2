import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms


class GroundedSAM2TestDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        """
        Args:
            image_dir (string): Path to the directory containing the test images.
            transform (callable, optional): A function/transform to apply to the images.
        """
        self.image_dir = image_dir
        self.image_files = sorted(os.listdir(image_dir))  # assuming images are sorted
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_name).convert("RGB")  # open and convert to RGB

        if self.transform:
            image = self.transform(image)

        return image, img_name

# Define image transformations
custom_transform = transforms.Compose([
    transforms.Resize((1024, 1024)),  # Resize images (you can change the size)
    transforms.ToTensor(),  # Convert PIL image to tensor
])

# Example directory with test images
#test_image_dir = "path/to/test/images"

# Create dataset instance
#test_dataset = GroundedSAM2TestDataset(image_dir=test_image_dir, transform=transform)