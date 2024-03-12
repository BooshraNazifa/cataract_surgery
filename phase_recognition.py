import os
import torch
from torchvision.io import read_image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Directory containing your images
image_dir = './frames'

# List of phase names as your classes
phases = ["Paracentesis", "Viscoelastic", "Wound", "Capsulorhexis", "Hydrodissection", 
          "Phaco", "Viscoelastic2", "IOL Insertion", "IOL Positioning", 
          "Viscoelastic removed", "Hydration", "Malyugin Ring Insertion", 
          "Malyugin Ring Removal", "Vision Blue"]

class SurgicalPhaseDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_labels = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]
    
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels[idx])
        image = read_image(img_path)
        label = self.img_labels[idx].split('_')[0]
        label = phases.index(label)  # Convert phase name to an integer index
        if self.transform:
            image = self.transform(image)
        return image, label

# Define transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create dataset
dataset = SurgicalPhaseDataset(image_dir, transform=transform)

# Print Dataset
for i, (image, label) in enumerate(dataset):
    print("Image shape:", image.shape, "| Label:", label)
    if i == 4:  # Print the first 5 items
        break

# Splitting dataset into train and test
train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=0.2, random_state=42)
train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)

# Create dataloaders
train_dataloader = DataLoader(dataset, batch_size=4, sampler=train_sampler)
val_dataloader = DataLoader(dataset, batch_size=4, sampler=val_sampler)

# Now you can iterate over the train_dataloader and val_dataloader for training and validation
