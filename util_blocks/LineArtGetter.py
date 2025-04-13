import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os


def extract_lineart_opencv(image_path, blur_kernel_size=5, canny_threshold1=100, canny_threshold2=200):
    """
    Extract line art from image using OpenCV

    Parameters:
        image_path: Input image path
        blur_kernel_size: Gaussian blur kernel size
        canny_threshold1: Lower threshold for Canny edge detection
        canny_threshold2: Upper threshold for Canny edge detection
    """
    # Check if file exists
    if isinstance(image_path, str) and not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Read image
    if isinstance(image_path, str):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read image file: {image_path}")
    else:
        image = image_path

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (blur_kernel_size, blur_kernel_size), 0)

    # Use Canny edge detection
    edges = cv2.Canny(blurred, canny_threshold1, canny_threshold2)

    # Invert colors to make lines black and background white
    lineart = 255 - edges

    return lineart, image


def main():
    # Set input image path with correct path format
    input_path = r"./img.png"

    # Check if file exists
    if not os.path.exists(input_path):
        print(f"Error: Image file not found '{input_path}'")
        print("Please ensure the file path is correct and the file exists")
        return

    try:
        # Extract line art
        lineart, original = extract_lineart_opencv(input_path)

        # Display results
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.title('origin')
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title('lineArt')
        plt.imshow(lineart, cmap='gray')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

        print("Line art extraction completed and displayed")
    except Exception as e:
        print(f"Error processing image: {e}")


if __name__ == "__main__":
    main()