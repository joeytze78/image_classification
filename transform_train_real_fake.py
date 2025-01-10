### extract real and fake image from train set (in Train_visualization folder)

import os
import shutil
import pandas as pd

# Path to the Train folder
train_folder = os.path.expanduser("~/CIDAUT/Train")
csv_file = os.path.expanduser("~/CIDAUT/train.csv")

# Load the CSV file
data = pd.read_csv(csv_file)

# Create "Real" and "Fake" folders
real_folder = os.path.join(train_folder, "Real")
fake_folder = os.path.join(train_folder, "Fake")
os.makedirs(real_folder, exist_ok=True)
os.makedirs(fake_folder, exist_ok=True)

# Move images to respective folders
for _, row in data.iterrows():
    image_name = row['image']
    label = row['label'].lower()  # Ensure label is lowercase
    
    src_path = os.path.join(train_folder, image_name)
    if label == 'real':
        dst_path = os.path.join(real_folder, image_name)
    elif label == 'editada':  # Assuming "editada" means "Fake"
        dst_path = os.path.join(fake_folder, image_name)
    else:
        print(f"Unknown label '{label}' for image {image_name}. Skipping.")
        continue
    
    # Move the file
    if os.path.exists(src_path):
        shutil.move(src_path, dst_path)
    else:
        print(f"File {src_path} does not exist. Skipping.")

print("Images organized into 'Real' and 'Fake' folders.")
