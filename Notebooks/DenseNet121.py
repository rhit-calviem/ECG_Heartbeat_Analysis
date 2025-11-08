# %%
# =========================================
# Optimized DenseNet121 Transfer Learning for 4-class ECG
# (Optimized for Quadro RTX 6000 GPUs)
# =========================================

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
from sklearn.metrics import confusion_matrix

# -------------------------------
# 0. GPU & Performance Settings
# -------------------------------a
if torch.cuda.is_available():
    # Auto-select least loaded GPU
    device = torch.device("cuda")
    print("✅ Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("⚠️ CUDA not available, using CPU")

torch.backends.cudnn.benchmark = True  # optimize conv kernels
torch.backends.cudnn.enabled = True

# -------------------------------
# 1. Dataset setup (same as before)
# -------------------------------
import kagglehub
os.environ["KAGGLE_CONFIG_DIR"] = os.getcwd()
dataset_path = kagglehub.dataset_download("evilspirit05/ecg-analysis")
train_dir = os.path.join(dataset_path, "ECG_DATA/train")
test_dir  = os.path.join(dataset_path, "ECG_DATA/test")

class ECGDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths, self.labels = [], []
        self.transform = transform
        class_map = {
            "ECG Images of Patient that have History of MI (172x12=2064)": 0,
            "ECG Images of Patient that have abnormal heartbeat (233x12=2796)": 1,
            "ECG Images of Myocardial Infarction Patients (240x12=2880)": 2,
            "Normal Person ECG Images (284x12=3408)": 3
        }
        for class_name, label in class_map.items():
            class_folder = os.path.join(root_dir, class_name)
            if not os.path.exists(class_folder): continue
            for file in os.listdir(class_folder):
                if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    self.image_paths.append(os.path.join(class_folder, file))
                    self.labels.append(label)

    def __len__(self): return len(self.image_paths)
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, self.labels[idx]

# -------------------------------
# 2. Transformations & DataLoader
# -------------------------------
imagenet_mean, imagenet_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
transform = transforms.Compose([
    transforms.Resize((160, 160)),  # smaller than 224x224 → faster
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0))
])

train_dataset = ECGDataset(train_dir, transform)
test_dataset  = ECGDataset(test_dir, transform)

subset_fraction = 0.20
random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

subset_size = max(1, int(subset_fraction * len(train_dataset)))
indices = random.sample(range(len(train_dataset)), subset_size)
subset_dataset = Subset(train_dataset, indices)
train_size = int(0.8 * len(subset_dataset))
val_size   = len(subset_dataset) - train_size
train_subset, val_subset = random_split(subset_dataset, [train_size, val_size])

batch_size = 64  # fits well on 24GB VRAM
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True)
val_loader   = DataLoader(val_subset,   batch_size=batch_size, shuffle=False, num_workers=6, pin_memory=True)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=6, pin_memory=True)

# -------------------------------
# 3. Model Setup (DenseNet121 + small head)
# -------------------------------
num_classes = 4
weights = models.DenseNet121_Weights.DEFAULT
model = models.densenet121(weights=weights)
in_features = model.classifier.in_features
model.classifier = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(in_features, 512),
    nn.ReLU(inplace=True),
    nn.Dropout(0.3),
    nn.Linear(512, num_classes)
)

# Freeze backbone for faster training
for name, param in model.named_parameters():
    if "classifier" not in name:
        param.requires_grad = True
print("Backbone frozen — training classifier head only.")

model = model.to(device)

# -------------------------------
# 4. Training Setup
# -------------------------------
criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

scaler = torch.cuda.amp.GradScaler()  # mixed precision

num_epochs = 25
train_losses, val_losses, train_accs, val_accs = [], [], [], []

# -------------------------------
# 5. Training Loop (with AMP)
# -------------------------------
for epoch in range(num_epochs):
    model.train()
    running_loss, correct, total = 0, 0, 0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_losses.append(running_loss / total)
    train_accs.append(correct / total)

    # Validation
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad(), torch.cuda.amp.autocast():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_losses.append(val_loss / val_total)
    val_accs.append(val_correct / val_total)
    scheduler.step()

    print(f"Epoch {epoch+1}: Train Acc={train_accs[-1]:.4f}, Val Acc={val_accs[-1]:.4f}")

    # -------------------------------
    # Early Stopping: Save best model
    # -------------------------------
    if epoch == 0:
        best_loss = val_losses[-1]  # initialize best_loss
    if val_losses[-1] < best_loss:
        best_loss = val_losses[-1]
        torch.save(model.state_dict(), 'best_model.pth')
        print(f"✅ Best model saved at epoch {epoch+1} with val_loss={best_loss:.4f}")

    if val_losses[-1] < best_loss:
        best_loss = val_losses[-1]
        torch.save(model.state_dict(), 'best_model.pth')
        print(f"✅ Best model saved at epoch {epoch+1} with val_loss={best_loss:.4f}")

# -------------------------------
# 6. Evaluation (Single Test Pass)
# -------------------------------
model.eval()
test_correct, test_total = 0, 0
all_preds, all_labels = [], []
with torch.no_grad(), torch.cuda.amp.autocast():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = outputs.max(1)
        test_correct += (preds == labels).sum().item()
        test_total += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print(f"✅ Test Accuracy: {test_correct/test_total:.4f}")

# -------------------------------
# 7. Confusion Matrix
# -------------------------------
cm = confusion_matrix(all_labels, all_preds)
class_names = ["History of MI", "Abnormal Heartbeat", "Myocardial Infarction", "Normal Person"]
plt.figure(figsize=(7,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("DenseNet121 Confusion Matrix on ECG Test Data")
plt.show()

import matplotlib.pyplot as plt

# After training loop, assuming you tracked loss values in a list called train_losses:
plt.figure(figsize=(8,5))
plt.plot(range(1, len(train_losses)+1), train_losses, marker='o')
plt.title('Training Loss vs. Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,5))
plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss', marker='o')
plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss', marker='s')
plt.title('Loss vs. Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



# %%
torch.save(model, 'best_DenseNet121_full.pth')



