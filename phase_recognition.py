import os
import torch
import pandas as pd
import numpy as np
from torchvision.io import read_image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix


# # Directory containing your images
# train_image_dir = './Files/TrainFrames'
# val_image_dir = './Files/ValFrames'
# test_image_dir = './Files/TestFrames'

# Directory containing your images in server
train_image_dir = '/scratch/booshra/50/TrainFrames'
val_image_dir = '/scratch/booshra/50/ValFrames'
test_image_dir = '/scratch/booshra/50/TestFrames'

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
        filename_content = self.img_labels[idx].split('_')
        label = filename_content[0]
        label = phases.index(label)
        if self.transform:
            image = self.transform(image)
        filename = filename_content[1]  
        timestamp = float(filename_content[-1].split('.')[0])  
        return image, label, filename, timestamp  
    
    

# Define transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



# Create dataset
print("Loading Train Dataset...")
train_dataset = SurgicalPhaseDataset(train_image_dir, transform=transform)
print("Loading Validation Dataset...")
val_dataset = SurgicalPhaseDataset(val_image_dir, transform=transform)
print("Loading Test Dataset...")
test_dataset = SurgicalPhaseDataset(test_image_dir, transform=transform)

# Create dataloaders
print("Creating Dataloaders...")
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=2)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=2)

# Load Pretrained ResNet and Modify Final Layer
resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
resnet.fc = nn.Linear(resnet.fc.in_features, 14)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = resnet.to(device)

# Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet.parameters(), lr=0.001)

# Initialize the scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

model_path = './model_checkpoint.pth'


if os.path.exists(model_path):
    resnet.load_state_dict(torch.load(model_path, map_location='cuda'))
    print("Model loaded successfully.")

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

    for inputs, labels, filename, timestamp in train_dataloader:
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
        for inputs, labels, filename, timestamp  in val_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
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
        checkpoint = {
            'model_state_dict': resnet.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss
        }
        torch.save(checkpoint, model_path)
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
all_preds = []
true_labels = []
results_data = []  # List to store data for final dataframe

with torch.no_grad():
    for inputs, labels, filenames, timestamps in test_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = resnet(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # Iterate over each sample in the batch
        for i in range(len(filenames)):
            filename = filenames[i]
            timestamp = timestamps[i].item()
            phase = phases[preds[i]]  
            true_label = phases[labels[i]]  
            
            # Append data to results list
            results_data.append({'Filename': filename, 'Timestamp': timestamp, 'Predicted Phase': phase, 'True Phase': true_label})
            all_preds.append(preds[i].item())
            true_labels.append(labels[i].item())
            

# Create DataFrame from results data
test_results_df = pd.DataFrame(results_data)
test_results_df.to_excel('test_results.xlsx', index=False)


precision = precision_score(true_labels, all_preds, average='weighted')
recall = recall_score(true_labels, all_preds, average='weighted')
f1 = f1_score(true_labels, all_preds, average='weighted')
test_accuracy = correct / total
conf_matrix = confusion_matrix(true_labels, all_preds)



# Print evaluation metrics
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

# Find the unique classes present in your test data
unique_classes = np.unique(true_labels + all_preds)

# Generate a list of phase names that match the unique classes found in the data
target_names_for_report = [phases[i] for i in unique_classes]

# Calculate and print classification report
report = classification_report(true_labels, all_preds, target_names=target_names_for_report)
print(report)


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
plt.savefig('./images/training_validation_test_performance.png')