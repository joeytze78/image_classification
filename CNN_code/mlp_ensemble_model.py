import pandas as pd

efficientnet_df = pd.read_csv("CNN_code/EfficientNetB0_pred.csv")
mnasnet_df = pd.read_csv("CNN_code/MNASNet_pred.csv")
laplacian_df = pd.read_csv("CNN_code/laplacian_pred.csv")

efficientnet_df["EfficientNet_Pred"] = efficientnet_df.apply(
    lambda row: "1" if row["real_score"] > row["fake_score"] else "0", axis=1
)
mnasnet_df["MNASNet_Pred"] = mnasnet_df.apply(
    lambda row: "1" if row["real_score"] > row["fake_score"] else "0", axis=1
)

# Merge all dataframes on the image name
ensemble_df = efficientnet_df[["image", "EfficientNet_Pred"]].merge(
    mnasnet_df[["image", "MNASNet_Pred"]], on="image"
).merge(
    laplacian_df[["Image Name", "Prediction"]], left_on="image", right_on="Image Name"
)

ensemble_df = ensemble_df.rename(columns={"Prediction": "Laplacian_Pred"})

# Convert Laplacian predictions to 1 for Real and 0 for Fake
ensemble_df["Laplacian_Pred"] = ensemble_df["Laplacian_Pred"].apply(lambda x: "1" if x == "Real" else "0")

# Perform majority voting
def majority_vote(row):
    votes = [row["EfficientNet_Pred"], row["MNASNet_Pred"], row["Laplacian_Pred"]]
    return max(set(votes), key=votes.count)

ensemble_df["label"] = ensemble_df.apply(majority_vote, axis=1)

final_output_path = "CNN_code/final_predictions.csv"
ensemble_df[["image", "label"]].to_csv(final_output_path, index=False)

print(f"Final predictions saved to: {final_output_path}")