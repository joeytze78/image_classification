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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.efficientnet_b0(weights=None)  
model.classifier = nn.Sequential(nn.Linear(model.classifier[1].in_features, 2))

model_weights_path = "model_path/run20/EfficientNetB0.pth"
model = model.to(device)


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

batch_size = 64
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

data_dir = "/home/joey/CIDAUT"
val_path = data_dir + "/Val_CNN"
val_dataset = AugmentedFolderDataset(root_dir=val_path, transform=data_transforms['val'], augmentations=None)
val_dataset_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

def generate_and_display_heatmaps(model, dataloader, target_layer, device, heatmap_output_folder, num_images=3):
    model.eval()  # Set the model to evaluation mode

    # Grad-CAM setup (no use_cuda argument)
    cam = GradCAM(model=model, target_layers=[target_layer])

    images_processed = 0
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1)  

        for i in range(inputs.size(0)):
            if images_processed >= num_images:
                return  # Stop after processing the desired number of images

            input_tensor = inputs[i].unsqueeze(0)
            label = torch.argmax(labels[i]).item()  
            pred_label = preds[i].item()

            # Generate heatmaps for Real (1) and Fake (0)
            heatmap_real = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(1)])
            heatmap_fake = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(0)])
            
            # Convert the image back to PIL format for visualization
            input_image = input_tensor.squeeze(0).cpu()
            input_image = F.to_pil_image(input_image)

            # Overlay heatmaps on the image
            heatmap_real_overlay = show_cam_on_image(np.array(input_image) / 255.0, heatmap_real[0], use_rgb=True)
            heatmap_fake_overlay = show_cam_on_image(np.array(input_image) / 255.0, heatmap_fake[0], use_rgb=True)

            # Plot the image and heatmaps
            plt.figure(figsize=(15, 5))
            plt.suptitle(f"Label: {label} (Ground Truth) | Prediction: {pred_label}", fontsize=16)

            # Original Image
            plt.subplot(1, 3, 1)
            plt.imshow(input_image)
            plt.title("Original Image")
            plt.axis("off")

            # Heatmap for Real
            plt.subplot(1, 3, 2)
            plt.imshow(heatmap_real_overlay)
            plt.title("Real Part Heatmap")
            plt.axis("off")

            # Heatmap for Fake
            plt.subplot(1, 3, 3)
            plt.imshow(heatmap_fake_overlay)
            plt.title("Fake Part Heatmap")
            plt.axis("off")

            # Show the plot
            plt.show()

            # Save heatmaps
            real_path = os.path.join(heatmap_output_folder, f"real_heatmap_{images_processed}_label_{label}_pred_{pred_label}.png")
            fake_path = os.path.join(heatmap_output_folder, f"fake_heatmap_{images_processed}_label_{label}_pred_{pred_label}.png")

            plt.imsave(real_path, heatmap_real_overlay)
            plt.imsave(fake_path, heatmap_fake_overlay)

            print(f"Saved real heatmap to: {real_path}")
            print(f"Saved fake heatmap to: {fake_path}")

            images_processed += 1

# generate Grad-CAM heatmaps
target_layer = model.features[-1]  # Target the last convolutional layer
run_folder = "CNN_code/"
heatmap_output_folder = os.path.join(run_folder, "EfficientNetB0/heatmaps")
os.makedirs(heatmap_output_folder, exist_ok=True)

generate_and_display_heatmaps(model, val_dataset_loader, target_layer, device, heatmap_output_folder, num_images=3)