import os
import torch
from torchvision import transforms
from PIL import Image
import pandas as pd
from torchvision import models

test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, test_dir, transform=None):
        self.test_dir = test_dir
        self.transform = transform
        self.image_paths = [
            os.path.join(test_dir, img) for img in os.listdir(test_dir)
            if img.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, os.path.basename(img_path)

test_dir = "/home/joey/CIDAUT/Test"
test_dataset = TestDataset(test_dir, transform=test_transforms)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.efficientnet_b0(weights=None)
model.classifier = torch.nn.Sequential(
    torch.nn.Linear(model.classifier[1].in_features, 1)
)
model = model.to(device)
model.load_state_dict(torch.load("/home/joey/CIDAUT/model_output/run11/EfficientNetB0.pth"))
model.eval()

# Run predictions on the test set
predictions = []

with torch.no_grad():
    for images, image_names in test_loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.sigmoid(outputs).cpu().numpy().flatten()
        preds = (probs > 0.5).astype(int)
        predictions.extend(zip(image_names, preds))

output_csv_path = "/home/joey/CIDAUT/first_submission.csv"
df = pd.DataFrame(predictions, columns=["image", "label"])
df.to_csv(output_csv_path, index=False)
print(f"Predictions saved to: {output_csv_path}")
