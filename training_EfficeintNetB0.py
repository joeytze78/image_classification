from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from torchvision import models
import torch
import torch.optim as optim
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import json
import imgaug.augmenters as iaa
from PIL import Image
import numpy as np

class AugmentedImageDataset(Dataset):
    """Custom Dataset for on-the-fly image loading and augmentation."""
    def __init__(self, image_folder, transform=None, augmentations=None):
        self.dataset = datasets.ImageFolder(image_folder)
        self.transform = transform
        self.augmentations = augmentations
        self.class_to_idx = self.dataset.class_to_idx
        self.classes = self.dataset.classes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]

        img = np.array(img)

        if self.augmentations:
            img = self.augmentations(image=img)

        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        return img, label

def train_model(model, dataloaders, criterion, optimizer, num_epochs):
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            total_samples = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device).float().unsqueeze(1)  # Ensure labels are [batch_size, 1]

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    preds = torch.argmax(outputs, dim=1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels)
                total_samples += inputs.size(0)

            epoch_loss = running_loss / total_samples
            epoch_acc = running_corrects.double() / total_samples

            if phase == 'train':
                train_loss_history.append(epoch_loss)
                train_acc_history.append(epoch_acc.item())
            else:
                val_loss_history.append(epoch_loss)
                val_acc_history.append(epoch_acc.item())

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

    return train_loss_history, train_acc_history, val_loss_history, val_acc_history

def save_plots(train_loss, val_loss, train_acc, val_acc, folder):
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label="Train Loss")
    plt.plot(val_loss, label="Val Loss")
    plt.title("Loss over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(folder, "loss_plot.png"))
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(train_acc, label="Train Accuracy")
    plt.plot(val_acc, label="Val Accuracy")
    plt.title("Accuracy over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(folder, "accuracy_plot.png"))
    plt.close()

def evaluate_and_save_metrics(model, dataloader, folder):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    conf_matrix = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=train_dataset.classes, yticklabels=train_dataset.classes)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    conf_matrix_path = os.path.join(folder, "confusion_matrix.png")
    plt.savefig(conf_matrix_path)
    plt.close()
    print(f"Confusion matrix saved as: {conf_matrix_path}")

def log_training_details_json(run_folder, model, optimizer, batch_size, num_epochs, learning_rate, criterion):
    log_data = {
        "Model": model.__class__.__name__,
        "Optimizer": optimizer.__class__.__name__,
        "Learning Rate": learning_rate,
        "Batch Size": batch_size,
        "Number of Epochs": num_epochs,
        "Loss Function": criterion.__class__.__name__,
        "Device": "GPU" if torch.cuda.is_available() else "CPU",
        "Augmentations": "Fliplr, Affine, GaussianBlur, AdditiveGaussianNoise during training when loading images",
        "Dataset Version" : "Version 2"
    }
    with open(os.path.join(run_folder, "training_details.json"), "w") as f:
        json.dump(log_data, f, indent=4)

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
}

train_augs = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Affine(rotate=(-10, 10),
               scale=(0.8, 1.2),
               shear=(-10, 10),
               translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}),
    iaa.GaussianBlur(sigma=(0, 1.0)),
    iaa.AdditiveGaussianNoise(scale=(0, 0.05*255)),
    # iaa.Multiply((0.8, 1.2)),
    # iaa.ContrastNormalization((0.8, 1.2)),
])

test_augs = iaa.Sequential([
    iaa.Resize((256, 256)),
])

data_dir = "/home/joey/CIDAUT"
train_path = data_dir + "/Train_CNN"
val_path = data_dir + "/Val_CNN"
train_dataset = AugmentedImageDataset(train_path, transform=data_transforms['train'], augmentations=train_augs)
val_dataset = AugmentedImageDataset(val_path, transform=data_transforms['val'], augmentations=test_augs)

batch_size = 64
lr = 0.001
num_epochs = 5

data_dir = "/home/joey/CIDAUT"

train_dataset_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print(train_dataset.classes)
print(val_dataset.classes)

num_classes = len(train_dataset.classes)
print(f"Classes: {train_dataset.classes}, number of classes: {num_classes}")

# Model setup
model = models.mnasnet1_3(weights="MNASNet1_3_Weights.IMAGENET1K_V1")
model.classifier = nn.Sequential(
    # nn.Linear(model.fc.in_features, 1), # for ResNet
    nn.Linear(model.classifier[1].in_features, 2),
    # nn.Sigmoid()  # used BCEWithLogitsLoss 
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()  # For binary classification
optimizer = optim.Adam(model.parameters(), lr=lr)

dataloaders = {"train": train_dataset_loader, "val": val_dataset_loader}

train_loss, train_acc, val_loss, val_acc = train_model(model, dataloaders, criterion, optimizer, num_epochs)

# Save model
output_folder = "/home/joey/CIDAUT/model_output"
os.makedirs(output_folder, exist_ok=True)

existing_runs = [folder_name for folder_name in os.listdir(output_folder) if folder_name.startswith("run")]
next_run_number = len(existing_runs) + 1
run_folder = os.path.join(output_folder, f"run{next_run_number}")
os.makedirs(run_folder, exist_ok=True)

# Save model state
model_path = os.path.join(run_folder, "EfficientNetB0.pth")
torch.save(model.state_dict(), model_path)
print(f"Model saved as: {model_path}")

save_plots(train_loss, val_loss, train_acc, val_acc, run_folder)

history_data = {
    "Epoch": list(range(1, num_epochs + 1)),
    "Train Loss": train_loss,
    "Val Loss": val_loss,
    "Train Accuracy": train_acc,
    "Val Accuracy": val_acc,
}
history_df = pd.DataFrame(history_data)
csv_path = os.path.join(run_folder, "training_history.csv")
history_df.to_csv(csv_path, index=False)
print(f"Training history saved as: {csv_path}")

evaluate_and_save_metrics(model, val_dataset_loader, run_folder)

log_training_details_json(run_folder, model, optimizer, batch_size, num_epochs, lr, criterion)