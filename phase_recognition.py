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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, precision_recall_curve


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
        filename_content = self.img_labels[idx].split('_')
        label = filename_content[0]
        label = phases.index(label)
        if self.transform:
            image = self.transform(image)
        filename = filename_content[1]  # Get the filename
        timestamp = float(filename_content[-1].split('.')[0])  # Assuming timestamp is the last component
        return image, label, filename, timestamp  # Return image, label, filename, and timestamp
    
    

# Define transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



# Create dataset
train_dataset = SurgicalPhaseDataset(train_image_dir, transform=transform)
val_dataset = SurgicalPhaseDataset(val_image_dir, transform=transform)
test_dataset = SurgicalPhaseDataset(test_image_dir, transform=transform)



# Create dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=True, num_workers=4)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=4)

# Load Pretrained ResNet and Modify Final Layer
resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
num_ftrs = resnet.fc.in_features
resnet.fc = nn.Sequential(
    nn.Dropout(0.5),  
    nn.Linear(num_ftrs, 14)  
)

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
all_preds = []
all_files = []
all_times = []

with torch.no_grad():
    for inputs, labels, filename, timestamp  in test_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = resnet(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        print(timestamp)

        # Store the predictions for this batch
        all_preds.extend(preds.cpu().numpy())
        all_files.extend(filename)
        all_times.extend(timestamp)

phase_predictions = [phases[p] for p in all_preds]
test_results_df = pd.DataFrame({'Filename': all_files,'Time Recorded':all_times, 'Phase': phase_predictions})
test_results_df.to_excel('test_results.xlsx', index=False)

# Just to display predictions:
for i, phase in enumerate(phase_predictions[:20]):  # Display first 20 predictions
    print(f"Frame {i}: Predicted phase - {phase}")


# Calculate evaluation metrics
test_labels = np.array([phases[label] for label in all_preds])
test_predictions = np.array([phases[label] for label in all_files])

precision = precision_score(test_labels, test_predictions, average='weighted')
recall = recall_score(test_labels, test_predictions, average='weighted')
f1 = f1_score(test_labels, test_predictions, average='weighted')
test_accuracy = correct / total


# Print evaluation metrics
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

# Confusion Matrix
conf_matrix = confusion_matrix(test_labels, test_predictions, labels=phases)
plt.figure(figsize=(10, 8))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(phases))
plt.xticks(tick_marks, phases, rotation=45)
plt.yticks(tick_marks, phases)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.savefig('./images/confusion_matrix.png')

# Calculate ROC curve and AUC
fpr, tpr, thresholds = roc_curve(test_labels, test_predictions, pos_label="Positive Class Label")
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.savefig('./images/roc_curve.png')

# Calculate precision-recall curve and AUC
precision, recall, thresholds = precision_recall_curve(test_labels, test_predictions, pos_label="Positive Class Label")
pr_auc = auc(recall, precision)

# Plot precision-recall curve
plt.figure()
plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve (area = %0.2f)' % pr_auc)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.savefig('./images/precision_recall_curve.png') 