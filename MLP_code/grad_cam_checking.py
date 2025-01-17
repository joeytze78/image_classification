import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn

# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        return self.softmax(x)

# Custom Grad-CAM for MLP
class GradCAMForMLP:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Hook to capture activations and gradients
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, input_tensor, target_class):
        self.model.eval()

        # Forward pass
        output = self.model(input_tensor)
        target = output[0, target_class]

        # Backward pass
        self.model.zero_grad()
        target.backward(retain_graph=True)

        # Compute Grad-CAM
        weights = self.gradients.mean(dim=0).detach().cpu().numpy()
        activations = self.activations[0].detach().cpu().numpy()
        cam = weights * activations
        cam = np.maximum(cam, 0)  # ReLU
        cam = cam / cam.max()  # Normalize
        return cam

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "/home/joey/CIDAUT/model_output/run21/mlp_model.pth"
model = MLP(input_dim=3, hidden_dim=64, output_dim=2)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval().to(device)

# Input example
input_sample = torch.tensor([0.5, 0.7, 0.2], dtype=torch.float32).unsqueeze(0).to(device)

# Target layer for Grad-CAM
target_layer = model.fc3

# Grad-CAM for MLP
cam = GradCAMForMLP(model=model, target_layer=target_layer)

# Generate Grad-CAM for "Realness" (class 0)
real_cam = cam(input_tensor=input_sample, target_class=0)

# Plot Grad-CAM heatmap for "Realness"
plt.figure(figsize=(6, 4))
plt.bar(range(len(real_cam)), real_cam, color="blue", alpha=0.7)
plt.xlabel("Feature Index")
plt.ylabel("Importance")
plt.title("Grad-CAM Heatmap for Realness")
plt.savefig("heatmap_real.png")
plt.show()

# Generate Grad-CAM for "Fakeness" (class 1)
fake_cam = cam(input_tensor=input_sample, target_class=1)

# Plot Grad-CAM heatmap for "Fakeness"
plt.figure(figsize=(6, 4))
plt.bar(range(len(fake_cam)), fake_cam, color="red", alpha=0.7)
plt.xlabel("Feature Index")
plt.ylabel("Importance")
plt.title("Grad-CAM Heatmap for Fakeness")
plt.savefig("heatmap_fake.png")
plt.show()
