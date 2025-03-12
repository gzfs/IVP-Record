import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import os

class ImageBlurrer:
    def __init__(self, image_path: str = "image.png"):
        """
        Initialize with an image.
        
        Args:
            image_path (str): Path to the input image
        """
        self.original = cv2.imread(image_path)
        if self.original is None:
            raise ValueError(f"Could not load image from {image_path}")
            
        self.original_rgb = cv2.cvtColor(self.original, cv2.COLOR_BGR2RGB)
        
        # Create output directory
        self.output_dir = "../output/blurring"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def apply_gaussian_blur(self, kernel_size: Tuple[int, int], sigma: float = 0) -> np.ndarray:
        """
        Apply Gaussian blur to the image.
        
        Args:
            kernel_size (tuple): Size of the Gaussian kernel
            sigma (float): Standard deviation of the Gaussian kernel
            
        Returns:
            numpy.ndarray: Blurred image
        """
        blurred = cv2.GaussianBlur(self.original, kernel_size, sigma)
        return cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)
    
    def apply_box_blur(self, kernel_size: Tuple[int, int]) -> np.ndarray:
        """
        Apply box (average) blur to the image.
        
        Args:
            kernel_size (tuple): Size of the averaging kernel
            
        Returns:
            numpy.ndarray: Blurred image
        """
        blurred = cv2.blur(self.original, kernel_size)
        return cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)
    
    def apply_motion_blur(self, kernel_size: int, angle: float) -> np.ndarray:
        """
        Apply motion blur to the image.
        
        Args:
            kernel_size (int): Size of the motion blur kernel
            angle (float): Angle of motion blur in degrees
            
        Returns:
            numpy.ndarray: Blurred image
        """
        # Create motion blur kernel
        kernel = np.zeros((kernel_size, kernel_size))
        center = kernel_size // 2
        
        # Convert angle to radians
        angle_rad = np.deg2rad(angle)
        
        # Create line using Bresenham's algorithm
        x = np.cos(angle_rad)
        y = np.sin(angle_rad)
        
        for i in range(kernel_size):
            offset = i - center
            x_pos = center + int(offset * x)
            y_pos = center + int(offset * y)
            if 0 <= x_pos < kernel_size and 0 <= y_pos < kernel_size:
                kernel[y_pos, x_pos] = 1
        
        # Normalize kernel
        kernel = kernel / kernel.sum()
        
        # Apply motion blur
        blurred = cv2.filter2D(self.original, -1, kernel)
        return cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)
    
    def apply_focus_blur(self, center: Tuple[int, int], radius: int, 
                        max_blur: int = 15) -> np.ndarray:
        """
        Apply focus blur (simulated depth of field) to the image.
        
        Args:
            center (tuple): Center point of focus
            radius (int): Radius of focused area
            max_blur (int): Maximum blur kernel size for unfocused areas
            
        Returns:
            numpy.ndarray: Blurred image with focus effect
        """
        height, width = self.original.shape[:2]
        y, x = np.ogrid[:height, :width]
        
        # Create distance mask from center
        dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        
        # Create normalized blur map
        blur_map = np.clip((dist_from_center - radius) / radius, 0, 1)
        
        # Initialize output image
        result = np.copy(self.original)
        
        # Apply varying blur based on distance
        for blur_size in range(3, max_blur + 1, 2):
            mask = (blur_map >= (blur_size-3)/(max_blur-3)) & (blur_map <= (blur_size-1)/(max_blur-3))
            if not np.any(mask):
                continue
                
            blurred = cv2.GaussianBlur(self.original, (blur_size, blur_size), 0)
            result[mask] = blurred[mask]
        
        return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    
    def compare_methods(self):
        """Compare different blur methods"""
        plt.figure(figsize=(15, 10))
        
        # Original image
        plt.subplot(231)
        plt.imshow(self.original_rgb)
        plt.title('Original')
        plt.axis('off')
        
        # Gaussian blur
        gaussian = self.apply_gaussian_blur((15, 15), 5)
        plt.subplot(232)
        plt.imshow(gaussian)
        plt.title('Gaussian Blur')
        plt.axis('off')
        
        # Box blur
        box = self.apply_box_blur((15, 15))
        plt.subplot(233)
        plt.imshow(box)
        plt.title('Box Blur')
        plt.axis('off')
        
        # Motion blur
        motion = self.apply_motion_blur(15, 45)
        plt.subplot(234)
        plt.imshow(motion)
        plt.title('Motion Blur')
        plt.axis('off')
        
        # Focus blur
        height, width = self.original.shape[:2]
        center = (width//2, height//2)
        focus = self.apply_focus_blur(center, radius=100)
        plt.subplot(235)
        plt.imshow(focus)
        plt.title('Focus Blur')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def analyze_blur_effects(self):
        """Analyze the effects of different blur methods"""
        methods = {
            'Gaussian': lambda: self.apply_gaussian_blur((15, 15), 5),
            'Box': lambda: self.apply_box_blur((15, 15)),
            'Motion': lambda: self.apply_motion_blur(15, 45)
        }
        
        # Calculate edge retention and smoothness for each method
        results = {}
        for name, blur_func in methods.items():
            blurred = blur_func()
            
            # Convert to grayscale for analysis
            gray_orig = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
            gray_blur = cv2.cvtColor(cv2.cvtColor(blurred, cv2.COLOR_RGB2BGR), 
                                   cv2.COLOR_BGR2GRAY)
            
            # Calculate edge retention
            edges_orig = cv2.Canny(gray_orig, 100, 200)
            edges_blur = cv2.Canny(gray_blur, 100, 200)
            edge_retention = np.sum(edges_blur) / np.sum(edges_orig)
            
            # Calculate smoothness (inverse of variance in local neighborhoods)
            smoothness = 1 / (np.var(gray_blur) + 1e-6)
            
            results[name] = {
                'Edge Retention': edge_retention,
                'Smoothness': smoothness
            }
        
        # Print results
        print("\nBlur Analysis Results:")
        for method, metrics in results.items():
            print(f"\n{method}:")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")

def main():
    try:
        # Example usage with default image
        blurrer = ImageBlurrer()
        
        # Compare different blur methods
        blurrer.compare_methods()
        
        # Save different blur effects
        # Gaussian blur with different kernel sizes
        for size in [5, 15, 31]:
            blurred = blurrer.apply_gaussian_blur((size, size), sigma=size/3)
            cv2.imwrite(f"{blurrer.output_dir}/gaussian_blur_{size}x{size}.jpg", 
                       cv2.cvtColor(blurred, cv2.COLOR_RGB2BGR))
        
        # Box blur with different kernel sizes
        for size in [5, 15, 31]:
            blurred = blurrer.apply_box_blur((size, size))
            cv2.imwrite(f"{blurrer.output_dir}/box_blur_{size}x{size}.jpg", 
                       cv2.cvtColor(blurred, cv2.COLOR_RGB2BGR))
        
        # Motion blur with different angles
        for angle in [0, 45, 90, 135]:
            blurred = blurrer.apply_motion_blur(15, angle)
            cv2.imwrite(f"{blurrer.output_dir}/motion_blur_{angle}deg.jpg", 
                       cv2.cvtColor(blurred, cv2.COLOR_RGB2BGR))
        
        # Focus blur with different radii
        height, width = blurrer.original.shape[:2]
        center = (width//2, height//2)
        for radius in [50, 100, 200]:
            blurred = blurrer.apply_focus_blur(center, radius)
            cv2.imwrite(f"{blurrer.output_dir}/focus_blur_r{radius}.jpg", 
                       cv2.cvtColor(blurred, cv2.COLOR_RGB2BGR))
        
        # Analyze blur effects
        blurrer.analyze_blur_effects()
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 