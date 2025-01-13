# CIDAUT AI Fake Scene Classification 2024

This repository contains the solution to the Kaggle competition **CIDAUT AI Fake Scene Classification 2024**. The goal is to classify images as either real or fake using machine learning techniques.

## Project Structure

```plaintext
.
├── .gitignore                 # List of files and directories to ignore in Git
├── cidaut_code.ipynb          # Jupyter notebook with exploratory data analysis and experiments
├── evaluation_code.py         # Code to evaluate the model on the test dataset
├── first_submission_ty.csv    # Example submission file for the competition
├── image_rgb_means.csv        # CSV file containing RGB mean values of images
├── preprocess_data_for_training.py # Script to preprocess data for training
├── README.md                  # Project documentation
├── requirements.txt           # Python dependencies
├── resnet50_training1.pth     # Pre-trained model file (ResNet50)
├── sample_submission.csv      # Kaggle sample submission format
├── temp_ela_image.jpg         # Temporary file used during analysis (e.g., ELA image)
├── train.csv                  # Training data CSV file
├── training1.py               # Main training script
├── transform_train_real_fake.py # Script for custom data transformations
```

## Getting Started

### Prerequisites

- Python 3.9 or higher
- Libraries as defined in `requirements.txt`

Install dependencies with:

```bash
pip install -r requirements.txt
```

### Running the Code

1. **Splitting Dataset**  
   Run `preprocess_data_for_training.py` to split the dataset for training. Train folder becomes 85% for train, 15% for val.
   
    - Train_CNN/Real: 344
    - Train_CNN/Fake: 267
    - Val_CNN/Real: 61
    - Val_CNN/Fake: 48

   ```bash
   python preprocess_data_for_training.py
   ```

2. **Training the Model**  
   Use `training1.py` to train the ResNet50 model.

   ```bash
   python training1.py
   ```

   The trained model will be saved as a `.pth` file in the `model_output` folder.

3. **Evaluating the Model**  
   Run `evaluation_code.py` to evaluate the trained model on the test dataset. A cv file will be generated for submission.

   ```bash
   python evaluation_code.py
   ```



## Key Files

- `training1.py`: Main script for training the model.
- `evaluation_code.py`: Script for testing the model and generating predictions.
- `first_submission_ty.csv`: EFirst submission.
- `transform_train_real_fake.py`: Split real and fake images in train folder according to train.csv .

## Competition Link

[CIDAUT AI Fake Scene Classification 2024](https://www.kaggle.com/competitions/cidaut-ai-fake-scene-classification-2024/overview)

## Notes

- The dataset is not included in this repository due to size and licensing constraints. Please download it from the Kaggle competition page.
- Dataset folder:
   ```
   Train (Original Train image dataset from Kaggel)
   Train_dataset (Processed Train folder)
   Train_CNN (Split from Train_dataset, train set)
   Valid_CNN (Split from Train_dataset, valid set)
   ```