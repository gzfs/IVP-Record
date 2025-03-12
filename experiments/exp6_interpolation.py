import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import os

class ImageInterpolator:
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
        self.output_dir = "../output/interpolation"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def resize_image(self, scale_factor: float, method: int) -> np.ndarray:
        """
        Resize image using specified interpolation method.
        
        Args:
            scale_factor (float): Scale factor for resizing
            method (int): OpenCV interpolation method
            
        Returns:
            numpy.ndarray: Resized image
        """
        if not scale_factor > 0:
            raise ValueError("Scale factor must be positive")
            
        height, width = self.original.shape[:2]
        new_height = int(height * scale_factor)
        new_width = int(width * scale_factor)
        
        resized = cv2.resize(self.original, (new_width, new_height), 
                           interpolation=method)
        return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    def compare_methods(self, scale_factor: float = 2.0):
        """
        Compare different interpolation methods.
        
        Args:
            scale_factor (float): Scale factor for resizing
        """
        methods = {
            'Nearest Neighbor': cv2.INTER_NEAREST,
            'Bilinear': cv2.INTER_LINEAR,
            'Bicubic': cv2.INTER_CUBIC,
            'Lanczos': cv2.INTER_LANCZOS4
        }
        
        plt.figure(figsize=(15, 10))
        
        # Original image
        plt.subplot(231)
        plt.imshow(self.original_rgb)
        plt.title('Original')
        plt.axis('off')
        
        # Interpolated versions
        for i, (name, method) in enumerate(methods.items(), 2):
            resized = self.resize_image(scale_factor, method)
            plt.subplot(230 + i)
            plt.imshow(resized)
            plt.title(f'{name} (x{scale_factor})')
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def analyze_quality(self, scale_up: float = 2.0, scale_down: Optional[float] = None):
        """
        Analyze quality metrics for different interpolation methods.
        
        Args:
            scale_up (float): Scale factor for upsampling
            scale_down (float, optional): Scale factor for downsampling first
        """
        methods = {
            'Nearest Neighbor': cv2.INTER_NEAREST,
            'Bilinear': cv2.INTER_LINEAR,
            'Bicubic': cv2.INTER_CUBIC,
            'Lanczos': cv2.INTER_LANCZOS4
        }
        
        # If scale_down is provided, first reduce the image
        if scale_down is not None:
            reference = self.resize_image(scale_down, cv2.INTER_AREA)
            target_size = self.original_rgb.shape[:2]
        else:
            reference = self.original_rgb
            target_size = tuple(int(x * scale_up) for x in self.original_rgb.shape[:2])
        
        results = {}
        for name, method in methods.items():
            # Resize to target size
            resized = cv2.resize(cv2.cvtColor(reference, cv2.COLOR_RGB2BGR),
                               target_size[::-1], interpolation=method)
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
            # Calculate metrics
            if scale_down is None:
                # For upscaling only, compare edge preservation
                edges_orig = cv2.Canny(cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY), 100, 200)
                edges_resized = cv2.Canny(cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY), 100, 200)
                edge_preservation = np.sum(edges_resized) / np.sum(edges_orig)
                results[name] = {'Edge Preservation': edge_preservation}
            else:
                # For down-then-up, compare with original
                mse = np.mean((self.original_rgb.astype(np.float32) - resized.astype(np.float32)) ** 2)
                psnr = 20 * np.log10(255.0) - 10 * np.log10(mse) if mse > 0 else float('inf')
                results[name] = {'MSE': mse, 'PSNR': psnr}
        
        # Print results
        print("\nQuality Metrics:")
        for method, metrics in results.items():
            print(f"\n{method}:")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.2f}")

def main():
    try:
        # Example usage with default image
        interpolator = ImageInterpolator()
        
        # Compare different methods (2x upscaling)
        interpolator.compare_methods(scale_factor=2.0)
        
        # Save interpolated images for different methods
        methods = {
            'nearest': cv2.INTER_NEAREST,
            'bilinear': cv2.INTER_LINEAR,
            'bicubic': cv2.INTER_CUBIC,
            'lanczos': cv2.INTER_LANCZOS4
        }
        
        for name, method in methods.items():
            # Save 2x upscaled version
            upscaled = interpolator.resize_image(2.0, method)
            cv2.imwrite(f"interpolation_{name}_2x.jpg", 
                       cv2.cvtColor(upscaled, cv2.COLOR_RGB2BGR))
            
            # Save 0.5x downscaled version
            downscaled = interpolator.resize_image(0.5, method)
            cv2.imwrite(f"interpolation_{name}_0.5x.jpg", 
                       cv2.cvtColor(downscaled, cv2.COLOR_RGB2BGR))
        
        # Analyze quality metrics
        print("\nUpscaling Analysis (2x):")
        interpolator.analyze_quality(scale_up=2.0)
        
        print("\nDown-then-Up Analysis (0.5x -> 2x):")
        interpolator.analyze_quality(scale_up=2.0, scale_down=0.5)
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 