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
from torch.utils.data import Dataset

class AugmentedFolderDataset(Dataset):
    """Custom Dataset for handling folder structure with augmentations."""
    def __init__(self, root_dir, transform=None, augmentations=None):
        self.data = []
        self.transform = transform
        self.augmentations = augmentations

        self.classes = ["Real", "Fake"]  
        
        real_dir = os.path.join(root_dir, "Real")
        fake_dir = os.path.join(root_dir, "Fake")

        # Real images (Real = 1, Fake = 0)
        for img_name in os.listdir(real_dir):
            self.data.append((os.path.join(real_dir, img_name), torch.tensor([1.0, 0.0], dtype=torch.float32)))

        # Fake images (Real = 0, Fake = 1)
        for img_name in os.listdir(fake_dir):
            self.data.append((os.path.join(fake_dir, img_name), torch.tensor([0.0, 1.0], dtype=torch.float32)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert("RGB")
        image_np = np.array(image)

        if self.augmentations:
            image_np = self.augmentations(image=image_np)

        image = Image.fromarray(image_np)

        if self.transform:
            image = self.transform(image)

        return image, label
    
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
            running_corrects = 0  # Initialize as integer
            total_samples = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    preds = torch.argmax(outputs, dim=1)  # Get predicted class (0 or 1)
                    targets = torch.argmax(labels, dim=1)  # Convert one-hot to class indices
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == targets).item()  # Sum correct predictions
                total_samples += inputs.size(0)

            epoch_loss = running_loss / total_samples
            epoch_acc = running_corrects / total_samples  # Calculate accuracy

            if phase == 'train':
                train_loss_history.append(epoch_loss)
                train_acc_history.append(epoch_acc)
            else:
                val_loss_history.append(epoch_loss)
                val_acc_history.append(epoch_acc)

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

    return train_loss_history, train_acc_history, val_loss_history, val_acc_history

def save_plots(train_loss, val_loss, train_acc, val_acc, folder):
    # Loss Plot
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label="Train Loss")
    plt.plot(val_loss, label="Val Loss")
    plt.title("Loss over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    loss_plot_path = os.path.join(folder, "loss_plot.png")
    plt.savefig(loss_plot_path)
    plt.close()

    # Accuracy Plot
    plt.figure(figsize=(10, 5))
    plt.plot(train_acc, label="Train Accuracy")
    plt.plot(val_acc, label="Val Accuracy")
    plt.title("Accuracy over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    acc_plot_path = os.path.join(folder, "accuracy_plot.png")
    plt.savefig(acc_plot_path)
    plt.close()

def evaluate_and_save_metrics(model, dataloader, folder):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            targets = torch.argmax(labels, dim=1)

            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)

            all_labels.extend(targets.cpu().numpy())
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

    # Classification Report
    class_report = classification_report(all_labels, all_preds, target_names=train_dataset.classes, output_dict=True)
    report_df = pd.DataFrame(class_report).transpose()
    report_path = os.path.join(folder, "classification_report.csv")
    report_df.to_csv(report_path, index=True)
    print(f"Classification report saved as: {report_path}")

    # Classification Report Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap="Blues")
    plt.title("Classification Report")
    plt.savefig(os.path.join(folder, "classification_report.png"))
    plt.close()
    print(f"Classification report heatmap saved as: {os.path.join(folder, 'classification_report.png')}")

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
    log_file = os.path.join(run_folder, "training_details.json")
    with open(log_file, "w") as f:
        json.dump(log_data, f, indent=4)
    print(f"Training details saved to: {log_file}")

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
train_dataset = AugmentedFolderDataset(root_dir=train_path, transform=data_transforms['train'], augmentations=train_augs)
val_dataset = AugmentedFolderDataset(root_dir=val_path, transform=data_transforms['val'], augmentations=None)
batch_size = 64
lr = 0.001
num_epochs = 200

data_dir = "/home/joey/CIDAUT"

train_dataset_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print(train_dataset.classes)
print(val_dataset.classes)

num_classes = len(train_dataset.classes)
print(f"Classes: {train_dataset.classes}, number of classes: {num_classes}")

# Model setup
# model = models.mnasnet1_3(weights="MNASNet1_3_Weights.IMAGENET1K_V1")
model = models.mnasnet0_5(weights="MNASNet0_5_Weights.IMAGENET1K_V1")
model.classifier = nn.Sequential(
    # nn.Linear(model.fc.in_features, 1), # for ResNet
    nn.Linear(model.classifier[1].in_features, 2),
    # nn.Sigmoid()  # used BCEWithLogitsLoss 
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.MSELoss()
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
model_path = os.path.join(run_folder, "MNASNet.pth")
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