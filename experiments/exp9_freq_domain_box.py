import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import os

class FrequencyDomainBoxFilter:
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
        self.output_dir = "../output/freq_box"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def create_box_filter(self, size: Tuple[int, int], 
                        cutoff_freq: float) -> np.ndarray:
        """
        Create a box filter in the frequency domain.
        
        Args:
            size (tuple): Size of the filter (height, width)
            cutoff_freq (float): Cutoff frequency (0 to 1)
            
        Returns:
            numpy.ndarray: Box filter in frequency domain
        """
        rows, cols = size
        crow, ccol = rows // 2, cols // 2
        
        # Create meshgrid for filter
        u = np.fft.fftfreq(cols)
        v = np.fft.fftfreq(rows)
        U, V = np.meshgrid(u, v)
        
        # Create box filter
        D = np.sqrt(U**2 + V**2)
        mask = D <= cutoff_freq
        
        # Save the filter visualization for debugging
        plt.figure(figsize=(8, 8))
        plt.imshow(mask, cmap='gray')
        plt.title(f'Box Filter (cutoff={cutoff_freq})')
        plt.colorbar()
        plt.savefig(os.path.join(self.output_dir, f'box_filter_mask_{cutoff_freq}.png'))
        plt.close()
        
        return mask.astype(np.float32)
    
    def apply_frequency_filter(self, cutoff_freq: float = 0.1) -> np.ndarray:
        """
        Apply box filter in frequency domain.
        
        Args:
            cutoff_freq (float): Cutoff frequency (0 to 1)
            
        Returns:
            numpy.ndarray: Filtered image
        """
        # Apply FFT
        f_transform = np.fft.fft2(self.original)
        f_shift = np.fft.fftshift(f_transform)
        
        # Create and apply filter
        box_filter = self.create_box_filter(self.original.shape, cutoff_freq)
        filtered_f_shift = f_shift * box_filter
        
        # Inverse FFT
        f_ishift = np.fft.ifftshift(filtered_f_shift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)
        
        # Normalize and enhance contrast for better visibility
        img_min = np.min(img_back)
        img_max = np.max(img_back)
        
        # Print debug info
        print(f"Box filter - Image min: {img_min}, max: {img_max}")
        
        # Normalize with enhanced contrast
        if img_max > img_min:
            img_back = (img_back - img_min) / (img_max - img_min)
        
        # Save the raw filtered image for debugging
        plt.figure(figsize=(10, 6))
        plt.imshow(img_back, cmap='gray')
        plt.title(f'Raw Filtered Image (cutoff={cutoff_freq})')
        plt.colorbar()
        plt.savefig(os.path.join(self.output_dir, f'raw_filtered_{cutoff_freq}.png'))
        plt.close()
        
        return (img_back * 255).astype(np.uint8)
    
    def save_plot(self, filename, plot_func):
        """Helper function to save plots"""
        plt.figure(figsize=(10, 4))
        plot_func()
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
    
    def visualize_frequency_domain(self, cutoff_freq: float = 0.1):
        """
        Save visualizations of the frequency domain representation and filtering process.
        
        Args:
            cutoff_freq (float): Cutoff frequency for the box filter
        """
        # Compute FFT
        f_transform = np.fft.fft2(self.original)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        
        # Create and apply filter
        box_filter = self.create_box_filter(self.original.shape, cutoff_freq)
        filtered_f_shift = f_shift * box_filter
        filtered_magnitude = np.log(np.abs(filtered_f_shift) + 1)
        
        # Apply inverse FFT
        filtered_image = self.apply_frequency_filter(cutoff_freq)
        
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
        
        # Save filtered spectrum
        def plot_filtered_spectrum():
            plt.imshow(filtered_magnitude, cmap='gray')
            plt.title('Filtered Spectrum')
            plt.axis('off')
        self.save_plot('filtered_spectrum.png', plot_filtered_spectrum)
        
        # Save filtered image
        cv2.imwrite(os.path.join(self.output_dir, 'box_filtered.jpg'), filtered_image)
        
        # Also save with matplotlib for comparison
        plt.figure(figsize=(10, 6))
        plt.imshow(filtered_image, cmap='gray')
        plt.title('Box Filtered Image (OpenCV)')
        plt.axis('off')
        plt.savefig(os.path.join(self.output_dir, 'box_filtered_plt.png'))
        plt.close()
    
    def analyze_frequency_response(self, cutoff_freqs: list):
        """
        Analyze the effect of different cutoff frequencies.
        
        Args:
            cutoff_freqs (list): List of cutoff frequencies to analyze
        """
        analysis_results = {}
        
        for freq in cutoff_freqs:
            filtered = self.apply_frequency_filter(freq)
            
            # Save filtered image for each frequency
            cv2.imwrite(os.path.join(self.output_dir, f'filtered_freq_{freq:.3f}.jpg'), filtered)
            
            # Also save with matplotlib for comparison
            plt.figure(figsize=(10, 6))
            plt.imshow(filtered, cmap='gray')
            plt.title(f'Box Filtered (cutoff={freq})')
            plt.axis('off')
            plt.savefig(os.path.join(self.output_dir, f'filtered_freq_{freq:.3f}_plt.png'))
            plt.close()
            
            # Analyze frequency content
            f_transform = np.fft.fft2(filtered)
            magnitude = np.abs(f_transform)
            
            analysis_results[freq] = {
                'avg_magnitude': float(np.mean(magnitude)),
                'max_magnitude': float(np.max(magnitude)),
                'content_retained': float(np.sum(magnitude)/np.sum(np.abs(np.fft.fft2(self.original))))
            }
        
        # Save analysis results
        np.save(os.path.join(self.output_dir, 'frequency_analysis.npy'), analysis_results)

def main():
    try:
        # Create filter processor
        filter_processor = FrequencyDomainBoxFilter()
        
        # Process and save frequency domain visualizations
        filter_processor.visualize_frequency_domain(cutoff_freq=0.2)  # Increased cutoff frequency
        
        # Analyze and save results for different cutoff frequencies
        cutoff_freqs = [0.05, 0.1, 0.2, 0.3, 0.5]  # Added higher cutoff frequency
        filter_processor.analyze_frequency_response(cutoff_freqs)
        
        print("Box filter processing completed successfully")
        
    except Exception as e:
        print(f"Error in box filter: {str(e)}")

if __name__ == "__main__":
    main() 