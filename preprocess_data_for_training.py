from sklearn.model_selection import train_test_split
import os
import shutil

def count_files(directory):
    return len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])

def split_data(source, train_dest, val_dest, valid_ratio=0.15):
    files = [f for f in os.listdir(source) if os.path.isfile(os.path.join(source, f))]
    
    train_files, val_files = train_test_split(files, test_size=valid_ratio, random_state=42)
    
    for file in train_files:
        shutil.copy(os.path.join(source, file), os.path.join(train_dest, file))
    for file in val_files:
        shutil.copy(os.path.join(source, file), os.path.join(val_dest, file))



source_folder = "/home/joey/CIDAUT/Train_visualization"
train_folder = "/home/joey/CIDAUT/Train_CNN"
val_folder = "/home/joey/CIDAUT/Val_CNN"

### Count before split
print("Train_visualization/Real: " + str(count_files(os.path.join(source_folder, "Real"))))
print("Train_visualization/Fake: " + str(count_files(os.path.join(source_folder, "Fake"))))

split_data(os.path.join(source_folder, "Fake"), os.path.join(train_folder, "Fake"), os.path.join(val_folder, "Fake"))
split_data(os.path.join(source_folder, "Real"), os.path.join(train_folder, "Real"), os.path.join(val_folder, "Real"))

print("Data successfully split into Train_CNN and Val_CNN folders.")

### Count after split
print("Train_CNN/Real: " + str(count_files(os.path.join(train_folder, "Real"))))
print("Train_CNN/Fake: " + str(count_files(os.path.join(train_folder, "Fake"))))
print("Val_CNN/Real: " + str(count_files(os.path.join(val_folder, "Real"))))
print("Val_CNN/Fake: " + str(count_files(os.path.join(val_folder, "Fake"))))

# Train_visualization/Real: 405
# Train_visualization/Fake: 315
# After Split:
# Train_CNN/Real: 344
# Train_CNN/Fake: 267
# Val_CNN/Real: 61
# Val_CNN/Fake: 48