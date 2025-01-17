import pandas as pd
import os

data_dir = "/home/joey/CIDAUT"
file_path = os.path.join(data_dir, 'CNN_code/test_images_blur_scores.csv')
data = pd.read_csv(file_path)

# Define threshold based on Laplacian Score for real (adjusted from train images --> see code in dataset_code_and_files/cidaut_code.ipynb)
laplacian_threshold = (200, 2000)

# Classification function using only Laplacian Score (because only laplacian score has different threshold for real and fake, other scores are too close)
def classify_image_laplacian(row, threshold):
    laplacian_score = row["Laplacian Score"]
    return "Real" if threshold[0] <= laplacian_score <= threshold[1] else "Fake"

data["Prediction"] = data.apply(classify_image_laplacian, threshold=laplacian_threshold, axis=1)

# data["Correct"] = ((data["Real"] == 1) & (data["Prediction"] == "Real")) | ((data["Fake"] == 1) & (data["Prediction"] == "Fake"))
# accuracy = data["Correct"].mean()

# print(f"Accuracy: {accuracy:.2f}")

output_path = os.path.join(data_dir, 'CNN_code/laplacian_pred.csv')
data[["Image Name", "Laplacian Score", "Prediction"]].to_csv(output_path, index=False)
print(f"Results saved to: {output_path}")