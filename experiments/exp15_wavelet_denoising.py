import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt
import os
from skimage.metrics import peak_signal_noise_ratio as psnr

class WaveletDenoiser:
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
        self.output_dir = "../output/wavelet"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def add_noise(self, mean: float = 0, std: float = 25) -> np.ndarray:
        """
        Add Gaussian noise to the image.
        
        Args:
            mean (float): Mean of the Gaussian noise
            std (float): Standard deviation of the Gaussian noise
            
        Returns:
            numpy.ndarray: Noisy image
        """
        noise = np.random.normal(mean, std, self.original.shape)
        noisy = self.original + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)
    
    def denoise_method1(self, noisy_img: np.ndarray, wavelet: str = 'haar') -> np.ndarray:
        """
        Denoise using method 1: Set all detail coefficients to zero.
        
        Args:
            noisy_img (numpy.ndarray): Input noisy image
            wavelet (str): Wavelet type
            
        Returns:
            numpy.ndarray: Denoised image
        """
        # Decompose
        coeffs = pywt.wavedec2(noisy_img, wavelet, level=1)
        cA, (cH, cV, cD) = coeffs
        
        # Set all detail coefficients to zero
        coeffs_modified = [cA, (np.zeros_like(cH), np.zeros_like(cV), np.zeros_like(cD))]
        
        # Reconstruct
        denoised = pywt.waverec2(coeffs_modified, wavelet)
        return np.clip(denoised, 0, 255).astype(np.uint8)
    
    def denoise_method2(self, noisy_img: np.ndarray, wavelet: str = 'haar', 
                       threshold: float = 30) -> np.ndarray:
        """
        Denoise using method 2: Apply thresholding to detail coefficients.
        
        Args:
            noisy_img (numpy.ndarray): Input noisy image
            wavelet (str): Wavelet type
            threshold (float): Threshold value for coefficient suppression
            
        Returns:
            numpy.ndarray: Denoised image
        """
        # Decompose
        coeffs = pywt.wavedec2(noisy_img, wavelet, level=1)
        cA, (cH, cV, cD) = coeffs
        
        # Apply thresholding to detail coefficients
        def threshold_coeff(c):
            return pywt.threshold(c, threshold, mode='soft')
        
        coeffs_modified = [cA, (threshold_coeff(cH), 
                               threshold_coeff(cV), 
                               threshold_coeff(cD))]
        
        # Reconstruct
        denoised = pywt.waverec2(coeffs_modified, wavelet)
        return np.clip(denoised, 0, 255).astype(np.uint8)
    
    def plot_coefficients(self, img: np.ndarray, title: str, filename: str):
        """
        Plot wavelet coefficients of an image.
        
        Args:
            img (numpy.ndarray): Input image
            title (str): Plot title
            filename (str): Output filename
        """
        coeffs = pywt.wavedec2(img, 'haar', level=1)
        cA, (cH, cV, cD) = coeffs
        
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        axes[0, 0].imshow(cA, cmap='gray')
        axes[0, 0].set_title('Approximation (LL)')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(cH, cmap='gray')
        axes[0, 1].set_title('Horizontal Detail (LH)')
        axes[0, 1].axis('off')
        
        axes[1, 0].imshow(cV, cmap='gray')
        axes[1, 0].set_title('Vertical Detail (HL)')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(cD, cmap='gray')
        axes[1, 1].set_title('Diagonal Detail (HH)')
        axes[1, 1].axis('off')
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()

def main():
    try:
        # Initialize denoiser
        denoiser = WaveletDenoiser()
        
        # Add noise to image
        noisy_img = denoiser.add_noise(std=25)
        
        # Apply both denoising methods
        denoised1 = denoiser.denoise_method1(noisy_img)
        denoised2 = denoiser.denoise_method2(noisy_img, threshold=30)
        
        # Plot coefficients for each stage
        denoiser.plot_coefficients(noisy_img, 'Wavelet Coefficients (Before Denoising)',
                                 'coefficients_before.png')
        denoiser.plot_coefficients(denoised1, 'Wavelet Coefficients (Method-1)',
                                 'coefficients_method1.png')
        denoiser.plot_coefficients(denoised2, 'Wavelet Coefficients (Method-2)',
                                 'coefficients_method2.png')
        
        # Compare results
        plt.figure(figsize=(15, 5))
        
        plt.subplot(141)
        plt.imshow(denoiser.original, cmap='gray')
        plt.title('Original')
        plt.axis('off')
        
        plt.subplot(142)
        plt.imshow(noisy_img, cmap='gray')
        plt.title(f'Noisy\nPSNR: {psnr(denoiser.original, noisy_img):.2f} dB')
        plt.axis('off')
        
        plt.subplot(143)
        plt.imshow(denoised1, cmap='gray')
        plt.title(f'Method-1\nPSNR: {psnr(denoiser.original, denoised1):.2f} dB')
        plt.axis('off')
        
        plt.subplot(144)
        plt.imshow(denoised2, cmap='gray')
        plt.title(f'Method-2\nPSNR: {psnr(denoiser.original, denoised2):.2f} dB')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(denoiser.output_dir, 'denoised_comparison.png'))
        plt.close()
        
        # Plot PSNR comparison
        methods = ['Noisy', 'Method-1', 'Method-2']
        psnr_values = [
            psnr(denoiser.original, noisy_img),
            psnr(denoiser.original, denoised1),
            psnr(denoiser.original, denoised2)
        ]
        
        plt.figure(figsize=(8, 5))
        plt.bar(methods, psnr_values)
        plt.title('PSNR Comparison')
        plt.ylabel('PSNR (dB)')
        plt.tight_layout()
        plt.savefig(os.path.join(denoiser.output_dir, 'psnr_comparison.png'))
        plt.close()
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 