# Loop through each validation path and generate heatmaps for both output folders
data_dir = "/home/joey/CIDAUT"
val_paths = ["Val_CNN/Real", "Val_CNN/Fake"]

from torchvision.transforms import functional as F
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import matplotlib.pyplot as plt
import os
import torch
import numpy as np
from torchvision import models
from torch import nn
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import csv
import cv2

# Custom dataset class
class AugmentedFolderDataset(Dataset):
    def __init__(self, img_dir, transform=None, augmentations=None):
        self.data = []
        self.transform = transform
        self.augmentations = augmentations

        for img_name in os.listdir(img_dir):
            self.data.append((os.path.join(img_dir, img_name), torch.tensor([0.0, 1.0], dtype=torch.float32)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert("RGB")

        if self.augmentations:
            image = self.augmentations(image=np.array(image))
            image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        return image, label, img_path

# Data setup
batch_size = 64
data_transforms = {
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
}

# Grad-CAM visualization
def generate_and_display_heatmaps(model, dataloader, target_layer, device, output_folder, num_images=None):
    model.eval()
    cam = GradCAM(model=model, target_layers=[target_layer])

    processed_images = 0
    all_img_paths = []  
    all_probabilities = [] 
    for inputs, labels, img_paths in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        all_img_paths.extend(img_paths)  
        all_probabilities.extend(probabilities.cpu().detach().numpy())
        print(f"Probabilities: {probabilities}")
        preds = torch.argmax(outputs, dim=1)

        for i in range(inputs.size(0)):
            if num_images is not None and processed_images >= num_images:
                break

            input_tensor = inputs[i].unsqueeze(0)
            label = torch.argmax(labels[i]).item()
            pred_label = preds[i].item()

            img_path = img_paths[i]
            img_name = os.path.basename(img_path)

            # Generate heatmaps
            if output_folder.endswith("real_heatmaps"):
                heatmap = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(1)])
            elif output_folder.endswith("fake_heatmaps"):
                heatmap = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(0)])
            # heatmap_real = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(1)])

            # Convert input image to PIL format
            input_image = F.to_pil_image(input_tensor.squeeze(0).cpu())

            # Real Heatmap  
            if output_folder.endswith("real_heatmaps"):
                heatmap_real_overlay = show_cam_on_image(np.array(input_image) / 255.0, heatmap[0], use_rgb=True)
                real_path = os.path.join(output_folder, f"real_heatmap_{img_name}_label_{label}_pred_{pred_label}.png")
                plt.imsave(real_path, heatmap_real_overlay)
                print(f"Saved real heatmap to: {real_path}") 
            elif output_folder.endswith("fake_heatmaps"):
                heatmap_fake_overlay = show_cam_on_image(np.array(input_image) / 255.0, heatmap[0], use_rgb=True)
                fake_path = os.path.join(output_folder, f"fake_heatmap_{img_name}_label_{label}_pred_{pred_label}.png")
                plt.imsave(fake_path, heatmap_fake_overlay)
                print(f"Saved fake heatmap to: {fake_path}")     
            
            processed_images += 1
    
    return all_img_paths, np.array(all_probabilities)

def calculate_blur_score(image_path):
    """
    Calculates the blurriness of an image using the variance of the Laplacian.
    A lower variance indicates higher blurriness.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None  # Return None if the image could not be read
    laplacian = cv2.Laplacian(image, cv2.CV_64F)  # Detect edges in the image
    return laplacian.var()


# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_configs = [
    {
        "model_class": models.efficientnet_b0,
        "weights_path": "/home/joey/CIDAUT/model_output/run20/EfficientNetB0.pth",
        "output_prefix": "EfficientNetB0"
    },
    {
        "model_class": models.mnasnet0_5,
        "weights_path": "/home/joey/CIDAUT/model_output/run21/MNASNet.pth",
        "output_prefix": "MNASNet"
    }
]

csv_data = [["Image Name", "Image Label", "EfficientNetB0 Real Probability", "EfficientNetB0 Fake Probability", "MNASNet Real Probability", "MNASNet Fake Probability", "Laplacian Score", "Laplacian Label"]]
output_folder_types = ["real_heatmaps", "fake_heatmaps"]
model_names = ["EfficientNetB0", "MNASNet"]
laplacian_threshold = (200, 2000)

for model_name in model_names:
    for output_folder_type in output_folder_types:
        # Construct the directory path
        directory_path = os.path.join(data_dir, model_name, output_folder_type)
        
        # Create the directory
        os.makedirs(directory_path, exist_ok=True)
        print(f"Directory created: {directory_path}")

for config in model_configs:
    model = config["model_class"](weights=None)
    model.classifier = nn.Sequential(nn.Linear(model.classifier[1].in_features, 2))
    model.load_state_dict(torch.load(config["weights_path"], map_location=device))
    model = model.to(device)
    model_name = config["output_prefix"]
    
    if model_name == "EfficientNetB0":
        target_layer=model.features[-1]
    elif model_name == "MNASNet":
        target_layer = model.layers[-1]

    for val_path in val_paths:
        img_dir = os.path.join(data_dir, val_path)
        val_dataset = AugmentedFolderDataset(img_dir=img_dir, transform=data_transforms['val'])
        val_dataset_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        for folder_type in output_folder_types:
            output_folder = os.path.join("eval_qualitative_findings", model_name, folder_type)
            os.makedirs(output_folder, exist_ok=True)

            print(f"Processing {val_path} to produce {folder_type}, saving heatmaps to {output_folder}")
            image_paths, probabilities = generate_and_display_heatmaps(
                model=model, 
                dataloader=val_dataset_loader, 
                target_layer=target_layer,
                device=device, 
                output_folder=output_folder
            )
        
        for img_path, prob in zip(image_paths, probabilities):
            img_name = os.path.basename(img_path)  
            path_parts = os.path.normpath(img_path).split(os.sep)
            if path_parts[-2] == "Real":
                image_label = 0 
            else:
                image_label = 1
            
            laplacian_score = calculate_blur_score(img_path)
            if laplacian_score is not None:
                laplacian_label = 0 if laplacian_threshold[0] <= laplacian_score <= laplacian_threshold[1] else 1
            else:
                laplacian_label = None

            # Find existing row or create a new one
            existing_row = next((row for row in csv_data if row[0] == img_name), None)
            if not existing_row:
                # Create a new row for this image
                new_row = [img_name, image_label, None, None, None, None, laplacian_score, laplacian_label]
                csv_data.append(new_row)
            else:
                new_row = existing_row
                new_row[6] = laplacian_score  # Update Laplacian score
                new_row[7] = laplacian_label

            if model_name == "EfficientNetB0":
                new_row[2] = prob[0]  # Real Probability
                new_row[3] = prob[1]  # Fake Probability
            elif model_name == "MNASNet":
                new_row[4] = prob[0]  # Real Probability
                new_row[5] = prob[1]  # Fake Probability

csv_output_path = os.path.join(data_dir, "eval_qualitative_findings", "combined_probabilities_output.csv")
with open(csv_output_path, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(csv_data)

print(f"Combined probabilities saved to {csv_output_path}")
