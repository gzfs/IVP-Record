import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

class IntensityReducer:
    def __init__(self, image_path: str = "image.png"):
        """
        Initialize with an image.
        
        Args:
            image_path (str): Path to the input image
        """
        self.original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if self.original is None:
            raise ValueError(f"Could not load image from {image_path}")
            
        # Create output directory
        self.output_dir = "../output/intensity"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def reduce_intensity(self, bits):
        """
        Reduce the intensity resolution to specified number of bits.
        
        Args:
            bits (int): Target bit depth (1-8)
            
        Returns:
            numpy.ndarray: Image with reduced intensity levels
        """
        if not 1 <= bits <= 8:
            raise ValueError("Bits must be between 1 and 8")
        
        # Calculate the scale factor
        levels = 2 ** bits
        scale = 255 / (levels - 1)
        
        # Quantize the image
        reduced = np.floor(self.original / scale) * scale
        return reduced.astype(np.uint8)
    
    def compare_bit_depths(self):
        """
        Compare different bit depth reductions.
        """
        plt.figure(figsize=(15, 10))
        
        # Original image (8-bit)
        plt.subplot(331)
        plt.imshow(self.original, cmap='gray')
        plt.title('Original (8-bit)')
        plt.axis('off')
        
        # Generate reduced versions
        for i, bits in enumerate(range(7, 0, -1), 2):
            reduced = self.reduce_intensity(bits)
            plt.subplot(330 + i)
            plt.imshow(reduced, cmap='gray')
            plt.title(f'{bits}-bit ({2**bits} levels)')
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "intensity_comparison.png"))
        plt.close()
    
    def analyze_histograms(self, bits):
        """
        Analyze and compare histograms of original and reduced images.
        
        Args:
            bits (int): Target bit depth for reduction
        """
        reduced = self.reduce_intensity(bits)
        
        plt.figure(figsize=(12, 5))
        
        # Original histogram
        plt.subplot(121)
        plt.hist(self.original.ravel(), bins=256, range=[0,256], density=True)
        plt.title('Original Histogram')
        plt.xlabel('Intensity Level')
        plt.ylabel('Frequency')
        
        # Reduced histogram
        plt.subplot(122)
        plt.hist(reduced.ravel(), bins=256, range=[0,256], density=True)
        plt.title(f'{bits}-bit Histogram')
        plt.xlabel('Intensity Level')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"histogram_comparison_{bits}bit.png"))
        plt.close()
        
        # Print statistics
        print(f"Original unique levels: {len(np.unique(self.original))}")
        print(f"Reduced unique levels: {len(np.unique(reduced))}")
        print(f"Mean absolute error: {np.mean(np.abs(self.original - reduced)):.2f}")

def main():
    try:
        # Example usage with default image
        reducer = IntensityReducer()
        
        # Compare different bit depths
        reducer.compare_bit_depths()
        
        # Analyze specific bit depth
        reducer.analyze_histograms(4)  # 4-bit example
        
        # Save reductions for different bit depths
        for bits in range(1, 9):
            reduced = reducer.reduce_intensity(bits)
            cv2.imwrite(os.path.join(reducer.output_dir, f"reduced_{bits}bit.jpg"), reduced)
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 