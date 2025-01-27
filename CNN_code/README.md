Training: 
1. run training_EfficientNetB0.py --> output: (pth file in model_output)
2. run training_MNAS.py --> output: (pth file in model_output)

Evaluation: 
1. run evaluation_code_EfficientNet.py, replace the EfficientNetB0.pth file --> output: EfficientNetB0_pred.csv
2. run evaluation_code_MNASNet.py, replace the EfficientNetB0.pth file --> output: MNASNet_pred.csv

Laplacian:
5. run create_csv_fft_laplacian_sobel.py --> output: test_images_blur_scores.csv (have laplacian score)
6. run laplacian_filter_pred.py --> output: laplacian_pred.csv
7. run mlp_ensemble_model.py --> output: final_predictions.csv