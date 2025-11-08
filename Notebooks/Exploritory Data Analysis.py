# %%
# %% [markdown]
# # ECG Image Dataset - Full EDA
# Downloads dataset from Kagglehub, loads using custom Dataset, and performs EDA.

# %%
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# %%
# --------------------------------------------
# 1. Download dataset from Kagglehub (if needed)
# --------------------------------------------
try:
    import kagglehub
except ImportError:
    !pip install kagglehub
    import kagglehub

# Set Kaggle config dir (where kaggle.json is located)
os.environ["KAGGLE_CONFIG_DIR"] = os.getcwd()

# Download dataset (will cache locally)
dataset_path = kagglehub.dataset_download("evilspirit05/ecg-analysis")
print("Dataset downloaded at:", dataset_path)

# Check folders
train_dir = os.path.join(dataset_path, "ECG_DATA", "train")
test_dir  = os.path.join(dataset_path, "ECG_DATA", "test")

print("Train folder exists:", os.path.exists(train_dir))
print("Test folder exists:", os.path.exists(test_dir))

# %%
# --------------------------------------------
# 2. Custom Dataset Class
# --------------------------------------------
class ECGDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform

        # Correct folder names from your dataset
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


# %%
# --------------------------------------------
# 3. Transforms & Load Dataset
# --------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_dataset = ECGDataset(train_dir, transform=transform)
test_dataset  = ECGDataset(test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

print(f"Train images: {len(train_dataset)}")
print(f"Test images:  {len(test_dataset)}")

# Combine datasets for EDA
full_dataset = train_dataset + test_dataset

# %%
# --------------------------------------------
# 4. Class Info & Distribution
# --------------------------------------------
class_names = ["History of MI", "abnormal heartbeat", "Myocardial Infarction", "Normal Person"]

def get_class_counts(dataset):
    counts = Counter([label for _, label in dataset])
    return [counts[i] for i in range(len(class_names))]

train_counts = get_class_counts(train_dataset)
test_counts  = get_class_counts(test_dataset)
total_counts = [t + v for t, v in zip(train_counts, test_counts)]

print("Class distributions (train + test):")
for cls, t, v in zip(class_names, train_counts, test_counts):
    print(f"{cls:25s} | Train: {t:5d} | Test: {v:5d} | Total: {t+v:5d}")

# Plot distribution
plt.figure(figsize=(8,5))
plt.bar(class_names, total_counts, color='teal', alpha=0.7)
plt.title("Overall Class Distribution")
plt.xlabel("Class")
plt.ylabel("Number of Images")
plt.show()

# %%
# --------------------------------------------
# 5. Image Statistics
# --------------------------------------------
means, stds, mins, maxs = [], [], [], []

for img, _ in tqdm(full_dataset, desc="Calculating image stats"):
    means.append(torch.mean(img))
    stds.append(torch.std(img))
    mins.append(torch.min(img))
    maxs.append(torch.max(img))

print("\nDataset Pixel Statistics:")
print(f"Mean pixel value: {np.mean(means):.4f}")
print(f"Std  pixel value: {np.mean(stds):.4f}")
print(f"Min pixel value:  {np.mean(mins):.4f}")
print(f"Max pixel value:  {np.mean(maxs):.4f}")

# %%
# --------------------------------------------
# 6. Visualize Sample Images
# --------------------------------------------
def show_samples(dataset, class_names, samples_per_class=4):
    fig, axs = plt.subplots(len(class_names), samples_per_class, figsize=(samples_per_class*3, len(class_names)*3))
    for i, class_name in enumerate(class_names):
        class_indices = [idx for idx, (_, label) in enumerate(dataset) if label == i]
        selected_indices = random.sample(class_indices, min(samples_per_class, len(class_indices)))
        for j, idx in enumerate(selected_indices):
            img, _ = dataset[idx]
            axs[i, j].imshow(np.transpose(img.numpy(), (1,2,0)), cmap='gray')
            axs[i, j].axis('off')
            if j == 0:
                axs[i, j].set_title(class_name, fontsize=10)
    plt.tight_layout()
    plt.show()

show_samples(full_dataset, class_names)



