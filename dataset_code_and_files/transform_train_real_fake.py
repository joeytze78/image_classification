### extract real and fake image from Train folder (without Real and Fake subfolders) to Train_dataset folder (with Real and Fake subfolders)

import os
import shutil
import pandas as pd

train_folder = os.path.expanduser("~/CIDAUT/Train")
csv_file = os.path.expanduser("~/CIDAUT/train_v2.csv")
data = pd.read_csv(csv_file)
label_counts = data['label'].value_counts()
print("Counts of each label in csv file:")
print(label_counts)

real_folder = os.path.expanduser("~/CIDAUT/Train_dataset/Real")
fake_folder = os.path.expanduser("~/CIDAUT/Train_dataset/Fake")
os.makedirs(real_folder, exist_ok=True)
os.makedirs(fake_folder, exist_ok=True)

# Move images to respective folders
for _, row in data.iterrows():
    image_name = row['image']
    label = row['label'].lower()  
    
    src_path = os.path.join(train_folder, image_name)
    if label == 'real':
        dst_path = os.path.join(real_folder, image_name)
    elif label == 'fake':  
        dst_path = os.path.join(fake_folder, image_name)
    else:
        print(f"Unknown label '{label}' for image {image_name}. Skipping.")
        continue
    
    if os.path.exists(src_path):
        shutil.copy2(src_path, dst_path)
    else:
        print(f"File {src_path} does not exist. Skipping.")

print("Images organized into 'Real' and 'Fake' folders.")

# Count number of files in real and fake folder
num_real_files = len([f for f in os.listdir(real_folder) if os.path.isfile(os.path.join(real_folder, f))])
num_fake_files = len([f for f in os.listdir(fake_folder) if os.path.isfile(os.path.join(fake_folder, f))])
print(f"Number of files in 'Real': {num_real_files}")
print(f"Number of files in 'Fake': {num_fake_files}")