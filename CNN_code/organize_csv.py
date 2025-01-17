import pandas as pd

# Load both CSV files
final_predictions = pd.read_csv("CNN_code/final_predictions.csv")
sample_submission = pd.read_csv("sample_submission.csv")

final_predictions_ordered = sample_submission[['image']].merge(final_predictions, on='image', how='left')

final_predictions_ordered.to_csv("CNN_code/final_predictions_reordered.csv", index=False)

print("The final_predictions.csv file has been reordered and saved as final_predictions_reordered.csv!")
