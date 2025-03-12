import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple
import os

class HistogramProcessor:
    def __init__(self, image_path: str = "image.png"):
        """
        Initialize with an image.
        
        Args:
            image_path (str): Path to the input image
        """
        self.original = cv2.imread(image_path)
        if self.original is None:
            raise ValueError(f"Could not load image from {image_path}")
            
        self.original_gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
        
        # Create output directory
        self.output_dir = "../output/histogram"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def calculate_histogram(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate histogram of grayscale image.
        
        Args:
            image (numpy.ndarray): Input grayscale image
            
        Returns:
            tuple: Histogram values and bin edges
        """
        hist, bins = np.histogram(image.flatten(), bins=256, range=[0, 256])
        return hist, bins
    
    def equalize_histogram(self, image: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Perform histogram equalization.
        
        Args:
            image (numpy.ndarray, optional): Input image (uses original if None)
            
        Returns:
            numpy.ndarray: Equalized image
        """
        if image is None:
            image = self.original_gray
            
        return cv2.equalizeHist(image)
    
    def match_histogram(self, target_image_path: str) -> np.ndarray:
        """
        Match histogram of original image to target image.
        
        Args:
            target_image_path (str): Path to target image
            
        Returns:
            numpy.ndarray: Image with matched histogram
        """
        # Read target image
        target = cv2.imread(target_image_path, cv2.IMREAD_GRAYSCALE)
        if target is None:
            raise ValueError("Could not load target image")
            
        # Calculate histograms
        src_hist, _ = self.calculate_histogram(self.original_gray)
        target_hist, _ = self.calculate_histogram(target)
        
        # Calculate cumulative distribution functions
        src_cdf = np.cumsum(src_hist) / src_hist.sum()
        target_cdf = np.cumsum(target_hist) / target_hist.sum()
        
        # Create lookup table
        lookup_table = np.zeros(256, dtype=np.uint8)
        j = 0
        for i in range(256):
            while j < 256 and target_cdf[j] < src_cdf[i]:
                j += 1
            lookup_table[i] = j
        
        # Apply lookup table
        return cv2.LUT(self.original_gray, lookup_table)
    
    def adaptive_histogram_equalization(self, clip_limit: float = 2.0, 
                                     grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
        """
        Perform adaptive histogram equalization (CLAHE).
        
        Args:
            clip_limit (float): Contrast limit for local histogram equalization
            grid_size (tuple): Size of grid for local processing
            
        Returns:
            numpy.ndarray: Enhanced image
        """
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
        return clahe.apply(self.original_gray)
    
    def analyze_histogram(self, image: np.ndarray, title: str = "Histogram"):
        """
        Analyze and display histogram of an image.
        
        Args:
            image (numpy.ndarray): Input image
            title (str): Title for the plot
        """
        hist, bins = self.calculate_histogram(image)
        
        # Calculate statistics
        mean = np.mean(image)
        std = np.std(image)
        median = np.median(image)
        
        # Plot histogram
        plt.figure(figsize=(10, 4))
        plt.plot(bins[:-1], hist)
        plt.title(f"{title}\nMean: {mean:.1f}, Std: {std:.1f}, Median: {median:.1f}")
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def compare_methods(self, target_image_path: Optional[str] = None):
        """
        Compare different histogram processing methods.
        
        Args:
            target_image_path (str, optional): Path to target image for histogram matching
        """
        # Prepare results
        results = {
            'Original': self.original_gray,
            'Histogram Equalization': self.equalize_histogram(),
            'Adaptive Equalization': self.adaptive_histogram_equalization()
        }
        
        if target_image_path:
            results['Histogram Matching'] = self.match_histogram(target_image_path)
        
        # Plot results
        n_images = len(results)
        cols = min(3, n_images)
        rows = (n_images + cols - 1) // cols
        
        plt.figure(figsize=(15, 5*rows))
        
        for i, (name, img) in enumerate(results.items(), 1):
            # Image
            plt.subplot(rows, cols, i)
            plt.imshow(img, cmap='gray')
            plt.title(name)
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Analyze histograms
        for name, img in results.items():
            self.analyze_histogram(img, name)

def main():
    try:
        # Example usage with default image
        processor = HistogramProcessor()
        
        # Standard histogram equalization
        equalized = processor.equalize_histogram()
        cv2.imwrite("histogram_equalized.jpg", equalized)
        
        # Adaptive histogram equalization with different parameters
        clip_limits = [2.0, 3.0, 4.0]
        grid_sizes = [(4, 4), (8, 8), (16, 16)]
        
        for clip_limit in clip_limits:
            for grid_size in grid_sizes:
                adaptive = processor.adaptive_histogram_equalization(
                    clip_limit=clip_limit, 
                    grid_size=grid_size
                )
                cv2.imwrite(
                    f"adaptive_hist_clip{int(clip_limit)}_grid{grid_size[0]}x{grid_size[1]}.jpg",
                    adaptive
                )
        
        # Try histogram matching if a reference image is available
        try:
            matched = processor.match_histogram("@reference.png")
            cv2.imwrite("histogram_matched.jpg", matched)
        except:
            print("Reference image not available for histogram matching")
        
        # Compare all methods
        processor.compare_methods()
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 