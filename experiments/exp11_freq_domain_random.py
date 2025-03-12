import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import os

class FrequencyDomainRandomFilter:
    def __init__(self, image_path: str = "image.png"):
        """
        Initialize with an image.
        
        Args:
            image_path (str): Path to the input image
        """
        self.original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if self.original is None:
            raise ValueError(f"Could not load image from {image_path}")
            
        # Convert to float32 for FFT
        self.original = self.original.astype(np.float32) / 255.0
        
        # Create output directory
        self.output_dir = "../output/freq_random"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def create_random_filter(self, size: Tuple[int, int], 
                           density: float = 0.5,
                           smoothness: float = 0.1) -> np.ndarray:
        """
        Create a random filter in the frequency domain.
        
        Args:
            size (tuple): Size of the filter (height, width)
            density (float): Density of random components (0 to 1)
            smoothness (float): Smoothness of the random filter
            
        Returns:
            numpy.ndarray: Random filter in frequency domain
        """
        rows, cols = size
        
        # Create base random filter
        np.random.seed(42)  # For reproducibility
        random_filter = np.random.rand(rows, cols)
        
        # Apply threshold based on density
        random_filter = (random_filter > (1 - density)).astype(np.float32)
        
        # Save the unsmoothed filter for debugging
        plt.figure(figsize=(8, 8))
        plt.imshow(random_filter, cmap='gray')
        plt.title(f'Random Filter Before Smoothing (density={density})')
        plt.colorbar()
        plt.savefig(os.path.join(self.output_dir, f'random_filter_unsmoothed_{density}.png'))
        plt.close()
        
        # Smooth the filter
        if smoothness > 0:
            random_filter = cv2.GaussianBlur(random_filter, 
                                           (0, 0), 
                                           smoothness * min(rows, cols))
        
        # Normalize
        random_filter = random_filter / random_filter.max()
        
        # Save the smoothed filter for debugging
        plt.figure(figsize=(8, 8))
        plt.imshow(random_filter, cmap='gray')
        plt.title(f'Random Filter After Smoothing (density={density}, smoothness={smoothness})')
        plt.colorbar()
        plt.savefig(os.path.join(self.output_dir, f'random_filter_smoothed_{density}_{smoothness}.png'))
        plt.close()
        
        return random_filter.astype(np.float32)
    
    def apply_frequency_filter(self, density: float = 0.5, 
                             smoothness: float = 0.1,
                             seed: Optional[int] = None) -> np.ndarray:
        """
        Apply random filter in frequency domain.
        
        Args:
            density (float): Density of random components
            smoothness (float): Smoothness of the random filter
            seed (int, optional): Random seed for reproducibility
            
        Returns:
            numpy.ndarray: Filtered image
        """
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Apply FFT
        f_transform = np.fft.fft2(self.original)
        f_shift = np.fft.fftshift(f_transform)
        
        # Create and apply filter
        random_filter = self.create_random_filter(self.original.shape, density, smoothness)
        filtered_f_shift = f_shift * random_filter
        
        # Inverse FFT
        f_ishift = np.fft.ifftshift(filtered_f_shift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)
        
        # Normalize and enhance contrast for better visibility
        img_min = np.min(img_back)
        img_max = np.max(img_back)
        
        # Print debug info
        print(f"Random filter - Image min: {img_min}, max: {img_max}")
        
        # Normalize with enhanced contrast
        if img_max > img_min:
            img_back = (img_back - img_min) / (img_max - img_min)
        
        # Save the raw filtered image for debugging
        plt.figure(figsize=(10, 6))
        plt.imshow(img_back, cmap='gray')
        plt.title(f'Raw Filtered Image (density={density}, smoothness={smoothness})')
        plt.colorbar()
        plt.savefig(os.path.join(self.output_dir, f'raw_filtered_{density}_{smoothness}.png'))
        plt.close()
        
        return (img_back * 255).astype(np.uint8)
    
    def save_plot(self, filename, plot_func):
        """Helper function to save plots"""
        plt.figure(figsize=(10, 4))
        plot_func()
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
    
    def visualize_frequency_domain(self, density: float = 0.5, 
                                 smoothness: float = 0.1,
                                 seed: Optional[int] = None):
        """
        Save visualizations of the frequency domain representation and filtering process.
        
        Args:
            density (float): Density of random components
            smoothness (float): Smoothness of the random filter
            seed (int, optional): Random seed for reproducibility
        """
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Compute FFT
        f_transform = np.fft.fft2(self.original)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        
        # Create and apply filter
        random_filter = self.create_random_filter(self.original.shape, density, smoothness)
        filtered_f_shift = f_shift * random_filter
        filtered_magnitude = np.log(np.abs(filtered_f_shift) + 1)
        
        # Apply inverse FFT
        filtered_image = self.apply_frequency_filter(density, smoothness, seed)
        
        # Save original image
        def plot_original():
            plt.imshow(self.original, cmap='gray')
            plt.title('Original Image')
            plt.axis('off')
        self.save_plot('original.png', plot_original)
        
        # Save frequency spectrum
        def plot_spectrum():
            plt.imshow(magnitude_spectrum, cmap='gray')
            plt.title('Frequency Spectrum')
            plt.axis('off')
        self.save_plot('spectrum.png', plot_spectrum)
        
        # Save random filter
        def plot_filter():
            plt.imshow(random_filter, cmap='gray')
            plt.title('Random Filter')
            plt.axis('off')
        self.save_plot('filter.png', plot_filter)
        
        # Save filtered spectrum
        def plot_filtered_spectrum():
            plt.imshow(filtered_magnitude, cmap='gray')
            plt.title('Filtered Spectrum')
            plt.axis('off')
        self.save_plot('filtered_spectrum.png', plot_filtered_spectrum)
        
        # Save filtered image
        cv2.imwrite(os.path.join(self.output_dir, f'random_filtered_d{int(density*100)}_s{int(smoothness*100)}.jpg'), filtered_image)
        
        # Also save with matplotlib for comparison
        plt.figure(figsize=(10, 6))
        plt.imshow(filtered_image, cmap='gray')
        plt.title(f'Random Filtered Image (density={density}, smoothness={smoothness})')
        plt.axis('off')
        plt.savefig(os.path.join(self.output_dir, f'random_filtered_plt.png'))
        plt.close()
        
        # Save filter profile
        def plot_profile():
            center_row = random_filter[random_filter.shape[0]//2, :]
            plt.plot(np.linspace(0, 1, len(center_row)), center_row)
            plt.title('Filter Profile (Center Row)')
            plt.xlabel('Normalized Position')
            plt.ylabel('Filter Value')
            plt.grid(True)
        self.save_plot('filter_profile.png', plot_profile)
    
    def analyze_parameter_effects(self, densities: list, smoothness_values: list):
        """
        Analyze the effect of different density and smoothness values.
        
        Args:
            densities (list): List of density values to analyze
            smoothness_values (list): List of smoothness values to analyze
        """
        analysis_results = {}
        
        for density in densities:
            for smoothness in smoothness_values:
                # Set a fixed seed for reproducibility
                np.random.seed(42)
                
                filtered = self.apply_frequency_filter(density, smoothness)
                
                # Save filtered image for each parameter combination
                cv2.imwrite(
                    os.path.join(self.output_dir, f'random_d{int(density*100)}_s{int(smoothness*100)}.jpg'), 
                    filtered
                )
                
                # Also save with matplotlib for comparison
                plt.figure(figsize=(10, 6))
                plt.imshow(filtered, cmap='gray')
                plt.title(f'Random Filtered (density={density}, smoothness={smoothness})')
                plt.axis('off')
                plt.savefig(os.path.join(self.output_dir, f'random_d{int(density*100)}_s{int(smoothness*100)}_plt.png'))
                plt.close()
                
                # Analyze frequency content
                f_transform = np.fft.fft2(filtered)
                magnitude = np.abs(f_transform)
                entropy = self.calculate_entropy(filtered)
                
                key = f"d{int(density*100)}_s{int(smoothness*100)}"
                analysis_results[key] = {
                    'avg_magnitude': float(np.mean(magnitude)),
                    'max_magnitude': float(np.max(magnitude)),
                    'content_retained': float(np.sum(magnitude)/np.sum(np.abs(np.fft.fft2(self.original)))),
                    'entropy': float(entropy)
                }
        
        # Save analysis results
        np.save(os.path.join(self.output_dir, 'parameter_analysis.npy'), analysis_results)
    
    @staticmethod
    def calculate_entropy(image: np.ndarray) -> float:
        """
        Calculate the entropy of an image.
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            float: Entropy value
        """
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist = hist / np.sum(hist)
        non_zero = hist > 0
        return -np.sum(hist[non_zero] * np.log2(hist[non_zero]))

def main():
    try:
        # Create filter processor
        filter_processor = FrequencyDomainRandomFilter()
        
        # Process and save frequency domain visualizations
        filter_processor.visualize_frequency_domain(density=0.3, smoothness=0.05, seed=42)  # Adjusted parameters
        
        # Analyze and save results for different parameter combinations
        densities = [0.1, 0.3, 0.5, 0.7]  # Different density values
        smoothness_values = [0.01, 0.05, 0.1, 0.2]  # Different smoothness values
        filter_processor.analyze_parameter_effects(densities, smoothness_values)
        
        print("Random filter processing completed successfully")
        
    except Exception as e:
        print(f"Error in random filter: {str(e)}")

if __name__ == "__main__":
    main() 