import os
import cv2
import numpy as np
import pandas as pd

def calculate_blur_score(image_path):
    """
    Calculates the blurriness of an image using the variance of the Laplacian.
    A lower variance indicates higher blurriness.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None  
    laplacian = cv2.Laplacian(image, cv2.CV_64F)  # detect edges in image
    return laplacian.var()

def calculate_fft_blur_score(image_path):
    """
    Calculates the FFT blur score of an image.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.log(np.abs(fshift) + 1)
    return np.mean(magnitude_spectrum)

def apply_sobel_filter(image_path):
    """
    Applies Sobel filter to an image to detect edges and calculates the mean gradient magnitude.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_magnitude = cv2.normalize(sobel_magnitude, None, 0, 255, cv2.NORM_MINMAX)
    return np.mean(sobel_magnitude)

def generate_blur_scores_csv(test_dir, output_csv_path):
    """
    Generates a CSV file containing blur scores and labels for images in the real and fake directories.

    Args:
        real_dir (str): Directory containing real images.
        fake_dir (str): Directory containing fake images.
        output_csv_path (str): Path to save the output CSV file.
    """
    data = {
        "Image Name": [],
        "FFT Score": [],
        "Laplacian Score": [],
        "Sobel Score": []
    }

    # Process real images
    for filename in os.listdir(test_dir):
        file_path = os.path.join(test_dir, filename)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            fft_score = calculate_fft_blur_score(file_path)
            laplacian_score = calculate_blur_score(file_path)
            sobel_score = apply_sobel_filter(file_path)
            if fft_score is not None and laplacian_score is not None and sobel_score is not None:
                data["Image Name"].append(filename)
                data["FFT Score"].append(fft_score)
                data["Laplacian Score"].append(laplacian_score)
                data["Sobel Score"].append(sobel_score)

    # Create DataFrame and save to CSV
    df = pd.DataFrame(data)
    df.to_csv(output_csv_path, index=False)
    print(f"CSV file saved at: {output_csv_path}")

test_dir = "/home/joey/CIDAUT/Test"
output_csv = "test_images_blur_scores.csv"
generate_blur_scores_csv(test_dir, output_csv)
