import os
import torch
from torchvision.io import read_image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Directory containing your images
train_image_dir = './TrainFrames'
val_image_dir = './ValFrames'
test_image_dir = './TestFrames'

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

# # Define transformations
# transform = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ConvertImageDtype(torch.float),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
    
# Transformations for the training set
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.Lambda(lambda x: x.float() / 255),  # Normalize to [0, 1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Transformations for the validation and test set
val_test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Lambda(lambda x: x.float() / 255),  # Normalize to [0, 1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Create dataset
train_dataset = SurgicalPhaseDataset(train_image_dir, transform=train_transform)
val_dataset = SurgicalPhaseDataset(val_image_dir, transform=val_test_transform)
test_dataset = SurgicalPhaseDataset(test_image_dir, transform=val_test_transform)

# Print train Dataset
for i, (image, label) in enumerate(train_dataset):
    print("Image shape:", image.shape, "| Label:", label)
    if i == 4:  # Print the first 5 items
        break

# Print val Dataset
for i, (image, label) in enumerate(val_dataset):
    print("Image shape:", image.shape, "| Label:", label)
    if i == 4:  # Print the first 5 items
        break

# Print test Dataset
for i, (image, label) in enumerate(test_dataset):
    print("Image shape:", image.shape, "| Label:", label)
    if i == 4:  # Print the first 5 items
        break



# Create dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=True, num_workers=4)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=4)

# Load Pretrained ResNet and Modify Final Layer
resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
resnet.fc = nn.Linear(resnet.fc.in_features, 15)

# Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet.parameters(), lr=0.001)

# Initialize the scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Check if CUDA is available, else use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move the model to the device
resnet.to(device)

train_losses, val_losses = [], []
train_accs, val_accs = [], []
best_val_loss = float('inf')

# Training Loop
for epoch in range(5):
    print(epoch)
    resnet.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = resnet(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    scheduler.step()
    train_losses.append(running_loss / len(train_dataloader))
    train_accs.append(correct / total)

    # Validation Loop
    resnet.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_dataloader:
            outputs = resnet(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_val_loss = val_loss / len(val_dataloader)
    epoch_val_acc = correct / total
    val_losses.append(epoch_val_loss)
    val_accs.append(epoch_val_acc)
    print(f'Val Loss: {epoch_val_loss:.4f}, Accuracy: {epoch_val_acc:.4f}')

    # Checkpointing
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        torch.save(resnet.state_dict(), 'model_checkpoint.pth')
        print('Validation loss decreased, saving checkpoint')

# Calculate overall accuracies
overall_train_accuracy = sum(train_accs) / len(train_accs)
overall_val_accuracy = sum(val_accs) / len(val_accs)

# Convert to percentage
overall_train_accuracy_percentage = overall_train_accuracy * 100
overall_val_accuracy_percentage = overall_val_accuracy * 100

print(f'Overall Training Accuracy: {overall_train_accuracy_percentage:.2f}%')
print(f'Overall Validation Accuracy: {overall_val_accuracy_percentage:.2f}%')


# Test Loop 
resnet.eval()  
test_loss = 0.0
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = resnet(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

test_accuracy = correct / total
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

phase_predictions = [phases[p] for p in preds.cpu().numpy()]  # Convert indices to phase names


# Just to display predictions:
for i, phase in enumerate(phase_predictions[:20]):  # Display first 20 predictions
    print(f"Frame {i}: Predicted phase - {phase}")

with open('phase_predictions.txt', 'w') as f:
    for i, phase in enumerate(phase_predictions):
        f.write(f"Frame {i}: {phase}\n")

# Plotting
plt.figure(figsize=(15, 5))  

plt.subplot(1, 3, 1)  # First subplot for training and validation loss
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.legend()
plt.title('Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(1, 3, 2)  # Second subplot for training and validation accuracy
plt.plot(train_accs, label='Training Accuracy')
plt.plot(val_accs, label='Validation Accuracy')
plt.legend()
plt.title('Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.subplot(1, 3, 3)  # Third subplot for test loss and accuracy
plt.bar(['Test Loss', 'Test Accuracy'], [test_loss / len(test_dataloader), test_accuracy])
plt.title('Test Performance')
plt.ylabel('Value')  # Modify as needed; might be percentage or raw number

plt.tight_layout()
plt.show()

plt.savefig('./images/training_validation_and_test_performance_10.png')
