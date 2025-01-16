import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

df = pd.read_csv("dataset_code_and_files/blur_scores_with_labels.csv")

class BlurDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe
        self.features = dataframe[['FFT Score', 'Laplacian Score', 'Sobel Score']].values
        self.labels = dataframe[['Real', 'Fake']].values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        labels = torch.tensor(self.labels[idx], dtype=torch.float32)
        return features, labels

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=1)  # For classification probabilities
        )

    def forward(self, x):
        return self.layers(x)
    
def save_training_results(run_folder, model, optimizer, batch_size, num_epochs, learning_rate, 
                          criterion, train_loss, val_loss, train_acc, val_acc, dataloader, class_names):
    os.makedirs(run_folder, exist_ok=True)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Log Training Details
    log_data = {
        "Model": model.__class__.__name__,
        "Optimizer": optimizer.__class__.__name__,
        "Learning Rate": learning_rate,
        "Batch Size": batch_size,
        "Number of Epochs": num_epochs,
        "Loss Function": criterion.__class__.__name__,
        "Device": "GPU" if torch.cuda.is_available() else "CPU",
        "Dataset Version": "Version 2"
    }
    log_file = os.path.join(run_folder, "training_details.json")
    with open(log_file, "w") as f:
        json.dump(log_data, f, indent=4)
    print(f"Training details saved to: {log_file}")

    # Save Training and Validation Plots
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_loss) + 1), train_loss, label="Train Loss", marker='o')
    plt.plot(range(1, len(val_loss) + 1), val_loss, label="Val Loss", marker='o')
    plt.title("Loss over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.xticks(range(1, len(train_loss) + 1))
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(run_folder, "loss_plot.png"))
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_acc) + 1), train_acc, label="Train Accuracy", marker='o')
    plt.plot(range(1, len(val_acc) + 1), val_acc, label="Val Accuracy", marker='o')
    plt.title("Accuracy over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.xticks(range(1, len(train_acc) + 1))
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(run_folder, "accuracy_plot.png"))
    plt.close()

    # Evaluate Metrics and Save Reports
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.to(device)
            outputs = model(features)
            preds = torch.argmax(outputs, dim=1)
            all_labels.extend(torch.argmax(labels, dim=1).cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # Confusion Matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    conf_matrix_path = os.path.join(run_folder, "confusion_matrix.png")
    plt.savefig(conf_matrix_path)
    plt.close()
    print(f"Confusion matrix saved as: {conf_matrix_path}")

    # Classification Report
    class_report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(class_report).transpose()
    report_path = os.path.join(run_folder, "classification_report.csv")
    report_df.to_csv(report_path, index=True)
    print(f"Classification report saved as: {report_path}")

    # Classification Report Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap="Blues")
    plt.title("Classification Report")
    plt.savefig(os.path.join(run_folder, "classification_report.png"))
    plt.close()
    print(f"Classification report heatmap saved as: {os.path.join(run_folder, 'classification_report.png')}")

    # Save Metrics to JSON
    metrics_data = {
        "Training Loss": train_loss,
        "Validation Loss": val_loss,
        "Training Accuracy": train_acc,
        "Validation Accuracy": val_acc,
        "Confusion Matrix": conf_matrix.tolist(),
        "Classification Report": class_report
    }
    metrics_file = os.path.join(run_folder, "metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics_data, f, indent=4)
    print(f"Metrics saved to: {metrics_file}")

# Hyperparameters
input_dim = 3  # Number of features: FFT, Laplacian, Sobel
hidden_dim = 64
output_dim = 2  # Real and Fake
batch_size = 32
learning_rate = 0.001
num_epochs = 5
class_names = ["Real", "Fake"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prepare dataset and dataloaders
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
train_dataset = BlurDataset(train_df)  
val_dataset = BlurDataset(val_df)

dataset = BlurDataset(df)
dataloader = {
    'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
    'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
}
model = MLP(input_dim, hidden_dim, output_dim).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_loss_history = []
val_loss_history = []
train_acc_history = []
val_acc_history = []

# Training and Validation loop
for epoch in range(num_epochs):
    # Training phase
    model.train()  
    total_train_loss = 0.0
    correct_train = 0
    total_train = 0
    
    for features, labels in dataloader['train']:
        features = features.to(device)
        labels = labels.to(device)

        # Convert one-hot labels to single class indices
        target = torch.argmax(labels, dim=1)
        
        # Forward pass
        outputs = model(features)
        loss = criterion(outputs, target)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        correct_train += (preds == target).sum().item()
        total_train += target.size(0)

    train_loss_history.append(total_train_loss / len(dataloader['train']))
    train_acc_history.append(correct_train / total_train * 100)
    
    print(f"\nEpoch [{epoch+1}/{num_epochs}]: \nTraining Loss: {total_train_loss:.4f}, Accuracy: {correct_train / total_train * 100:.2f}%")

    # Validation phase
    model.eval()  
    total_val_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():  # No gradients needed for validation
        for features, labels in dataloader['val']:
            features = features.to(device)
            labels = labels.to(device)
            target = torch.argmax(labels, dim=1)
            
            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, target)
            
            total_val_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct_val += (preds == target).sum().item()
            total_val += target.size(0)

    val_loss_history.append(total_val_loss / len(dataloader['val']))
    val_acc_history.append(correct_val / total_val * 100)
    
    print(f"Validation Loss: {total_val_loss:.4f}, Accuracy: {correct_val / total_val * 100:.2f}%")

# Save model
output_folder = "/home/joey/CIDAUT/model_output"
os.makedirs(output_folder, exist_ok=True)

existing_runs = [folder_name for folder_name in os.listdir(output_folder) if folder_name.startswith("run")]
next_run_number = len(existing_runs) + 1
run_folder = os.path.join(output_folder, f"run{next_run_number}")
os.makedirs(run_folder, exist_ok=True)

model_path = os.path.join(run_folder, "mlp_model.pth")
save_training_results(
    run_folder=run_folder,
    model=model,
    optimizer=optimizer,
    batch_size=batch_size,
    num_epochs=num_epochs,
    learning_rate=learning_rate,
    criterion=criterion,
    train_loss=train_loss_history,
    val_loss=val_loss_history,
    train_acc=train_acc_history,
    val_acc=val_acc_history,
    dataloader=dataloader['val'],  
    class_names=["Real", "Fake"]
)

torch.save(model.state_dict(), model_path)
print(f"Model saved to: {model_path}")