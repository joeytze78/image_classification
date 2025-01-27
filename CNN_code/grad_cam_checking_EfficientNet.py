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

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model setup
model = models.efficientnet_b0(weights=None)
model.classifier = nn.Sequential(nn.Linear(model.classifier[1].in_features, 2))
model_weights_path = "/home/joey/CIDAUT/model_output/run20/EfficientNetB0.pth"
model.load_state_dict(torch.load(model_weights_path, map_location=device))
model = model.to(device)

# Custom dataset class
class AugmentedFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None, augmentations=None):
        self.data = []
        self.transform = transform
        self.augmentations = augmentations

        # real_dir = os.path.join(root_dir, "Real")
        # for img_name in os.listdir(real_dir):
        #     self.data.append((os.path.join(real_dir, img_name), torch.tensor([1.0, 0.0], dtype=torch.float32)))

        fake_dir = os.path.join(root_dir, "Fake")
        for img_name in os.listdir(fake_dir):
            self.data.append((os.path.join(fake_dir, img_name), torch.tensor([0.0, 1.0], dtype=torch.float32)))

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

data_dir = "/home/joey/CIDAUT"
val_path = os.path.join(data_dir, "Val_CNN")
val_dataset = AugmentedFolderDataset(root_dir=val_path, transform=data_transforms['val'])
val_dataset_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Grad-CAM visualization
def generate_and_display_heatmaps(model, dataloader, target_layer, device, output_folder, num_images=3):
    model.eval()
    cam = GradCAM(model=model, target_layers=[target_layer])

    processed_images = 0
    for inputs, labels, img_paths in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        print(f"Probabilities: {probabilities}")
        preds = torch.argmax(outputs, dim=1)

        for i in range(inputs.size(0)):
            if processed_images >= num_images:
                return

            input_tensor = inputs[i].unsqueeze(0)
            label = torch.argmax(labels[i]).item()
            pred_label = preds[i].item()

            img_path = img_paths[i]
            img_name = os.path.basename(img_path)

            # Generate heatmaps
            heatmap_real = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(1)])
            # heatmap_fake = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(0)])

            # Convert input image to PIL format
            input_image = F.to_pil_image(input_tensor.squeeze(0).cpu())

            # Fake Heatmap
            # heatmap_fake_overlay = show_cam_on_image(np.array(input_image) / 255.0, heatmap_fake[0], use_rgb=True)
            # fake_path = os.path.join(output_folder, f"fake_heatmap_{img_name}_label_{label}_pred_{pred_label}.png")
            # plt.imsave(fake_path, heatmap_fake_overlay)
            # print(f"Saved fake heatmap to: {fake_path}")

            # Real Heatmap  
            heatmap_real_overlay = show_cam_on_image(np.array(input_image) / 255.0, heatmap_real[0], use_rgb=True)
            real_path = os.path.join(output_folder, f"real_heatmap_{img_name}_label_{label}_pred_{pred_label}.png")
            plt.imsave(real_path, heatmap_real_overlay)
            print(f"Saved real heatmap to: {real_path}")       
            
            processed_images += 1

# Generate Grad-CAM heatmaps
target_layer = model.features[-1]
output_folder = "CNN_code/EfficientNetB0/real_heatmaps"
os.makedirs(output_folder, exist_ok=True)

generate_and_display_heatmaps(model, val_dataset_loader, target_layer, device, output_folder, num_images=3)
