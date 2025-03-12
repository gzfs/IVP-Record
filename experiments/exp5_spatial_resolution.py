import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

class SpatialResolutionModifier:
    def __init__(self, image_path: str = "image.png"):
        """
        Initialize with an image.
        
        Args:
            image_path (str): Path to the input image
        """
        self.original = cv2.imread(image_path)
        if self.original is None:
            raise ValueError(f"Could not load image from {image_path}")
            
        # Convert BGR to RGB for display
        self.original_rgb = cv2.cvtColor(self.original, cv2.COLOR_BGR2RGB)
        
        # Create output directory
        self.output_dir = "../output/spatial"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def reduce_resolution(self, scale_factor):
        """
        Reduce spatial resolution by a scale factor.
        
        Args:
            scale_factor (float): Factor to reduce resolution by (e.g., 0.5 for half)
            
        Returns:
            numpy.ndarray: Reduced resolution image
        """
        if not 0 < scale_factor <= 1:
            raise ValueError("Scale factor must be between 0 and 1")
            
        height, width = self.original.shape[:2]
        new_height = int(height * scale_factor)
        new_width = int(width * scale_factor)
        
        # Reduce resolution
        reduced = cv2.resize(self.original, (new_width, new_height))
        
        # Scale back to original size for comparison
        restored = cv2.resize(reduced, (width, height))
        
        return cv2.cvtColor(restored, cv2.COLOR_BGR2RGB)
    
    def compare_resolutions(self, scale_factors):
        """
        Compare different resolution reductions.
        
        Args:
            scale_factors (list): List of scale factors to compare
        """
        n = len(scale_factors) + 1  # +1 for original
        cols = min(3, n)
        rows = (n + cols - 1) // cols
        
        plt.figure(figsize=(15, 5*rows))
        
        # Original image
        plt.subplot(rows, cols, 1)
        plt.imshow(self.original_rgb)
        plt.title('Original')
        plt.axis('off')
        
        # Reduced versions
        for i, factor in enumerate(scale_factors, 2):
            reduced = self.reduce_resolution(factor)
            plt.subplot(rows, cols, i)
            plt.imshow(reduced)
            plt.title(f'Scale Factor: {factor:.2f}')
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def analyze_quality(self, scale_factor):
        """
        Analyze quality metrics for a specific resolution reduction.
        
        Args:
            scale_factor (float): Scale factor for reduction
        """
        reduced = self.reduce_resolution(scale_factor)
        
        # Calculate MSE
        mse = np.mean((self.original_rgb.astype(np.float32) - reduced.astype(np.float32)) ** 2)
        
        # Calculate PSNR
        if mse == 0:
            psnr = float('inf')
        else:
            psnr = 20 * np.log10(255.0) - 10 * np.log10(mse)
        
        print(f"Quality Metrics for scale factor {scale_factor}:")
        print(f"Mean Squared Error (MSE): {mse:.2f}")
        print(f"Peak Signal-to-Noise Ratio (PSNR): {psnr:.2f} dB")
        
        # Display comparison
        plt.figure(figsize=(12, 6))
        
        plt.subplot(121)
        plt.imshow(self.original_rgb)
        plt.title('Original')
        plt.axis('off')
        
        plt.subplot(122)
        plt.imshow(reduced)
        plt.title(f'Reduced (Scale: {scale_factor:.2f})')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

def main():
    try:
        # Example usage with default image
        modifier = SpatialResolutionModifier()
        
        # Save original image at 100%
        cv2.imwrite(os.path.join(modifier.output_dir, "spatial_reduced_100percent.jpg"), 
                   cv2.cvtColor(modifier.original_rgb, cv2.COLOR_RGB2BGR))
        
        # Compare different resolutions
        scale_factors = [0.75, 0.5, 0.25]
        modifier.compare_resolutions(scale_factors)
        
        # Save reduced versions
        for factor in scale_factors:
            reduced = modifier.reduce_resolution(factor)
            cv2.imwrite(os.path.join(modifier.output_dir, f"spatial_reduced_{int(factor*100)}percent.jpg"), 
                       cv2.cvtColor(reduced, cv2.COLOR_RGB2BGR))
        
        # Analyze specific resolution
        modifier.analyze_quality(0.5)
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 