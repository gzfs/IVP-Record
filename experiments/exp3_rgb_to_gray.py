import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

class RGBToGrayscaleConverter:
    def __init__(self, image_path: str = "image.png"):
        """
        Initialize with an image.
        
        Args:
            image_path (str): Path to the input image
        """
        """Initialize the converter with standard luminance coefficients"""
        self.coefficients = {
            'red': 0.299,
            'green': 0.587,
            'blue': 0.114
        }
        
        # Create output directory
        self.output_dir = "../output/grayscale"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def convert(self, image_path, custom_coefficients=None):
        """
        Convert RGB image to grayscale using luminance method.
        
        Args:
            image_path (str): Path to the input RGB image
            custom_coefficients (dict, optional): Custom RGB coefficients
        
        Returns:
            numpy.ndarray: Grayscale image
        """
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
            
        # Split into channels
        b, g, r = cv2.split(image)
        
        # Use custom coefficients if provided
        if custom_coefficients:
            r_weight = custom_coefficients['red']
            g_weight = custom_coefficients['green']
            b_weight = custom_coefficients['blue']
        else:
            r_weight = self.coefficients['red']
            g_weight = self.coefficients['green']
            b_weight = self.coefficients['blue']
        
        # Apply luminance formula
        grayscale = r_weight * r + g_weight * g + b_weight * b
        
        # Convert to uint8
        return grayscale.astype(np.uint8)
    
    def compare_methods(self, image_path):
        """
        Compare custom grayscale conversion with OpenCV's method.
        
        Args:
            image_path (str): Path to the input RGB image
        """
        # Custom conversion
        custom_gray = self.convert(image_path)
        
        # OpenCV's conversion
        image = cv2.imread(image_path)
        opencv_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate difference
        diff = cv2.absdiff(custom_gray, opencv_gray)
        max_diff = np.max(diff)
        avg_diff = np.mean(diff)
        
        print(f"Maximum difference between implementations: {max_diff/255:.2f}")
        print(f"Average difference between implementations: {avg_diff/255:.2f}")
        
        # Create comparison visualization
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(custom_gray, cmap='gray')
        plt.title('Custom Implementation')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(opencv_gray, cmap='gray')
        plt.title('OpenCV Implementation')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(diff, cmap='hot')
        plt.title('Difference (Amplified)')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "grayscale_comparison.png"))
        plt.close()

def main():
    try:
        converter = RGBToGrayscaleConverter()
        
        # Example usage with default image
        image_path = "image.png"
        
        # Standard conversion
        converter.compare_methods(image_path)
        
        # Save standard grayscale conversion
        custom_gray = converter.convert(image_path)
        cv2.imwrite(os.path.join(converter.output_dir, "grayscale_custom.jpg"), custom_gray)
        
        # OpenCV's conversion for comparison
        image = cv2.imread(image_path)
        opencv_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(os.path.join(converter.output_dir, "grayscale_opencv.jpg"), opencv_gray)
        
        # Custom coefficients example
        custom_coeffs = {
            'red': 0.33,
            'green': 0.56,
            'blue': 0.11
        }
        custom_gray = converter.convert(image_path, custom_coeffs)
        cv2.imwrite(os.path.join(converter.output_dir, "grayscale_custom_coeffs.jpg"), custom_gray)
        
        plt.figure(figsize=(5, 5))
        plt.imshow(custom_gray, cmap='gray')
        plt.title('Custom Coefficients Result')
        plt.axis('off')
        plt.savefig(os.path.join(converter.output_dir, "grayscale_custom_coeffs_plot.png"))
        plt.close()
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 