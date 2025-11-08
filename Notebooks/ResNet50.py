# %%
# =========================================
# Full ECG ResNet50 Training + EDA Script
# =========================================

import os
import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models

# -------------------------------
# 1. Download Dataset from Kagglehub
# -------------------------------
import kagglehub

# Set Kaggle config directory
os.environ["KAGGLE_CONFIG_DIR"] = os.getcwd()

dataset_path = kagglehub.dataset_download("evilspirit05/ecg-analysis")
print("Dataset downloaded at:", dataset_path)

train_dir = os.path.join(dataset_path, "ECG_DATA/train")
test_dir  = os.path.join(dataset_path, "ECG_DATA/test")

print("Train folder exists:", os.path.exists(train_dir))
print("Test folder exists:", os.path.exists(test_dir))

# -------------------------------
# 2. Define Custom Dataset
# -------------------------------
class ECGDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform

        # Folder names from the dataset
        class_map = {
            "ECG Images of Patient that have History of MI (172x12=2064)": 0,
            "ECG Images of Patient that have abnormal heartbeat (233x12=2796)": 1,
            "ECG Images of Myocardial Infarction Patients (240x12=2880)": 2,
            "Normal Person ECG Images (284x12=3408)": 3
        }

        for class_name, label in class_map.items():
            class_folder = os.path.join(root_dir, class_name)
            if not os.path.exists(class_folder):
                print(f"Warning: folder not found -> {class_folder}")
                continue
            for file in os.listdir(class_folder):
                if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    self.image_paths.append(os.path.join(class_folder, file))
                    self.labels.append(label)

        print(f"Found {len(self.image_paths)} images in {root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# -------------------------------
# 3. Transforms & Load Dataset
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_dataset = ECGDataset(train_dir, transform=transform)
test_dataset  = ECGDataset(test_dir, transform=transform)

# Quick train/val split
train_size = int(0.8 * len(train_dataset))
val_size   = len(train_dataset) - train_size
train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_subset, batch_size=32, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

# -------------------------------
# 4. Exploratory Data Analysis
# -------------------------------
# Class distribution
train_labels = [label for _, label in train_dataset]
unique, counts = np.unique(train_labels, return_counts=True)
print("Class distribution in training set:")
for u, c in zip(unique, counts):
    print(f"Class {u}: {c} images")

# Display random samples
fig, axes = plt.subplots(1, 4, figsize=(12,4))
for i, class_idx in enumerate(unique):
    class_imgs = [img for img, label in train_dataset if label == class_idx]
    img = transforms.ToPILImage()(class_imgs[random.randint(0, len(class_imgs)-1)])
    axes[i].imshow(img)
    axes[i].set_title(f"Class {class_idx}")
    axes[i].axis('off')
plt.show()

# -------------------------------
# 5. Setup ResNet50 Model
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

num_classes = 4
model = models.resnet50(weights=None)  # Start from scratch
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# -------------------------------
# 6. Training Loop
# -------------------------------
num_epochs = 1  # increase as needed
train_losses, val_losses = [], []
train_accs, val_accs = [], []

for epoch in range(num_epochs):
    # Train
    model.train()
    running_loss, correct, total = 0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    epoch_loss = running_loss / total
    epoch_acc  = correct / total
    train_losses.append(epoch_loss)
    train_accs.append(epoch_acc)

    # Validate
    model.eval()
    val_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    val_epoch_loss = val_loss / total
    val_epoch_acc  = correct / total
    val_losses.append(val_epoch_loss)
    val_accs.append(val_epoch_acc)

    print(f"Epoch {epoch+1}/{num_epochs} | "
          f"Train Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f} | "
          f"Val Loss: {val_epoch_loss:.4f}, Acc: {val_epoch_acc:.4f}")

# -------------------------------
# 7. Plot Training Curves
# -------------------------------
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs+1), val_losses, label='Val Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training & Validation Loss")
plt.legend()

plt.subplot(1,2,2)
plt.plot(range(1, num_epochs+1), train_accs, label='Train Acc')
plt.plot(range(1, num_epochs+1), val_accs, label='Val Acc')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training & Validation Accuracy")
plt.legend()

plt.show()



