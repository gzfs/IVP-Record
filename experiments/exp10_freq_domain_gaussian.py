import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import os

class FrequencyDomainGaussianFilter:
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
        self.output_dir = "../output/freq_gaussian"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def create_gaussian_filter(self, size: Tuple[int, int], 
                             sigma: float) -> np.ndarray:
        """
        Create a Gaussian filter in the frequency domain.
        
        Args:
            size (tuple): Size of the filter (height, width)
            sigma (float): Standard deviation of the Gaussian filter
            
        Returns:
            numpy.ndarray: Gaussian filter in frequency domain
        """
        rows, cols = size
        crow, ccol = rows // 2, cols // 2
        
        # Create meshgrid for filter
        u = np.fft.fftfreq(cols)
        v = np.fft.fftfreq(rows)
        U, V = np.meshgrid(u, v)
        
        # Create Gaussian filter
        D = np.sqrt(U**2 + V**2)
        gaussian = np.exp(-D**2 / (2 * sigma**2))
        
        # Save the filter visualization for debugging
        plt.figure(figsize=(8, 8))
        plt.imshow(gaussian, cmap='gray')
        plt.title(f'Gaussian Filter (sigma={sigma})')
        plt.colorbar()
        plt.savefig(os.path.join(self.output_dir, f'gaussian_filter_vis_{sigma}.png'))
        plt.close()
        
        return gaussian.astype(np.float32)
    
    def apply_frequency_filter(self, sigma: float = 0.1) -> np.ndarray:
        """
        Apply Gaussian filter in frequency domain.
        
        Args:
            sigma (float): Standard deviation of the Gaussian filter
            
        Returns:
            numpy.ndarray: Filtered image
        """
        # Apply FFT
        f_transform = np.fft.fft2(self.original)
        f_shift = np.fft.fftshift(f_transform)
        
        # Create and apply filter
        gaussian_filter = self.create_gaussian_filter(self.original.shape, sigma)
        filtered_f_shift = f_shift * gaussian_filter
        
        # Inverse FFT
        f_ishift = np.fft.ifftshift(filtered_f_shift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)
        
        # Normalize and enhance contrast for better visibility
        img_min = np.min(img_back)
        img_max = np.max(img_back)
        
        # Print debug info
        print(f"Gaussian filter - Image min: {img_min}, max: {img_max}")
        
        # Normalize with enhanced contrast
        if img_max > img_min:
            img_back = (img_back - img_min) / (img_max - img_min)
        
        # Save the raw filtered image for debugging
        plt.figure(figsize=(10, 6))
        plt.imshow(img_back, cmap='gray')
        plt.title(f'Raw Filtered Image (sigma={sigma})')
        plt.colorbar()
        plt.savefig(os.path.join(self.output_dir, f'raw_filtered_{sigma}.png'))
        plt.close()
        
        return (img_back * 255).astype(np.uint8)
    
    def save_plot(self, filename, plot_func):
        """Helper function to save plots"""
        plt.figure(figsize=(10, 4))
        plot_func()
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
    
    def visualize_frequency_domain(self, sigma: float = 0.1):
        """
        Save visualizations of the frequency domain representation and filtering process.
        
        Args:
            sigma (float): Standard deviation for the Gaussian filter
        """
        # Compute FFT
        f_transform = np.fft.fft2(self.original)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        
        # Create and apply filter
        gaussian_filter = self.create_gaussian_filter(self.original.shape, sigma)
        filtered_f_shift = f_shift * gaussian_filter
        filtered_magnitude = np.log(np.abs(filtered_f_shift) + 1)
        
        # Apply inverse FFT
        filtered_image = self.apply_frequency_filter(sigma)
        
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
        
        # Save Gaussian filter
        def plot_filter():
            plt.imshow(gaussian_filter, cmap='gray')
            plt.title('Gaussian Filter')
            plt.axis('off')
        self.save_plot('filter.png', plot_filter)
        
        # Save filtered spectrum
        def plot_filtered_spectrum():
            plt.imshow(filtered_magnitude, cmap='gray')
            plt.title('Filtered Spectrum')
            plt.axis('off')
        self.save_plot('filtered_spectrum.png', plot_filtered_spectrum)
        
        # Save filtered image
        cv2.imwrite(os.path.join(self.output_dir, 'gaussian_filtered_sigma10.jpg'), filtered_image)
        
        # Also save with matplotlib for comparison
        plt.figure(figsize=(10, 6))
        plt.imshow(filtered_image, cmap='gray')
        plt.title('Gaussian Filtered Image (OpenCV)')
        plt.axis('off')
        plt.savefig(os.path.join(self.output_dir, 'gaussian_filtered_plt.png'))
        plt.close()
        
        # Save filter response plot
        def plot_response():
            center_row = gaussian_filter[gaussian_filter.shape[0]//2, :]
            plt.plot(np.linspace(-0.5, 0.5, len(center_row)), center_row)
            plt.title('Filter Response')
            plt.xlabel('Normalized Frequency')
            plt.ylabel('Magnitude')
            plt.grid(True)
        self.save_plot('filter_response.png', plot_response)
    
    def analyze_frequency_response(self, sigmas: list):
        """
        Analyze the effect of different sigma values.
        
        Args:
            sigmas (list): List of sigma values to analyze
        """
        analysis_results = {}
        
        for sigma in sigmas:
            filtered = self.apply_frequency_filter(sigma)
            
            # Save filtered image for each sigma
            cv2.imwrite(os.path.join(self.output_dir, f'gaussian_filtered_sigma{int(sigma*100)}.jpg'), filtered)
            
            # Also save with matplotlib for comparison
            plt.figure(figsize=(10, 6))
            plt.imshow(filtered, cmap='gray')
            plt.title(f'Gaussian Filtered (sigma={sigma})')
            plt.axis('off')
            plt.savefig(os.path.join(self.output_dir, f'gaussian_filtered_sigma{int(sigma*100)}_plt.png'))
            plt.close()
            
            # Analyze frequency content
            f_transform = np.fft.fft2(filtered)
            magnitude = np.abs(f_transform)
            smoothness = 1 / (np.var(filtered) + 1e-6)
            
            analysis_results[sigma] = {
                'avg_magnitude': float(np.mean(magnitude)),
                'max_magnitude': float(np.max(magnitude)),
                'content_retained': float(np.sum(magnitude)/np.sum(np.abs(np.fft.fft2(self.original)))),
                'smoothness': float(smoothness)
            }
        
        # Save analysis results
        np.save(os.path.join(self.output_dir, 'frequency_analysis.npy'), analysis_results)

def main():
    try:
        # Create filter processor
        filter_processor = FrequencyDomainGaussianFilter()
        
        # Process and save frequency domain visualizations
        filter_processor.visualize_frequency_domain(sigma=0.05)  # Decreased sigma for more visible effect
        
        # Analyze and save results for different sigma values
        sigmas = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]  # Added more sigma values
        filter_processor.analyze_frequency_response(sigmas)
        
        print("Gaussian filter processing completed successfully")
        
    except Exception as e:
        print(f"Error in Gaussian filter: {str(e)}")

if __name__ == "__main__":
    main() 