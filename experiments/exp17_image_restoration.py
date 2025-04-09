import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

class ImageRestorer:
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
        self.output_dir = "../output/image_restoration"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Resize to 256x256 and normalize
        self.original = cv2.resize(self.original, (256, 256))
        self.original = self.original / 255.0
    
    def create_gaussian_kernel(self, kernel_size=15, sigma=3):
        """Create a Gaussian blur kernel"""
        ax = np.linspace(-(kernel_size-1)/2, (kernel_size-1)/2, kernel_size)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        return kernel / np.sum(kernel)
    
    def degrade_image(self, kernel, noise_level=0.01):
        """Degrade the image using blur and noise"""
        # Apply blur
        blurred = cv2.filter2D(self.original, -1, kernel)
        
        # Add noise
        noise = np.random.normal(0, noise_level, self.original.shape)
        degraded = blurred + noise
        degraded = np.clip(degraded, 0, 1)
        
        return degraded, noise
    
    def inverse_filter(self, degraded_img, kernel, threshold=0.01):
        """Apply inverse filtering"""
        # Pad kernel
        img_height, img_width = degraded_img.shape
        k_height, k_width = kernel.shape
        
        padded_kernel = np.zeros((img_height, img_width))
        start_h = (img_height - k_height) // 2
        start_w = (img_width - k_width) // 2
        padded_kernel[start_h:start_h+k_height, start_w:start_w+k_width] = kernel
        padded_kernel = np.fft.ifftshift(padded_kernel)
        
        # Compute FFTs
        degraded_fft = np.fft.fft2(degraded_img)
        kernel_fft = np.fft.fft2(padded_kernel)
        
        # Apply threshold
        kernel_fft_thresholded = kernel_fft.copy()
        kernel_fft_thresholded[np.abs(kernel_fft) < threshold] = threshold
        
        # Inverse filtering
        restored_fft = degraded_fft / kernel_fft_thresholded
        restored_fft = np.fft.ifftshift(restored_fft)
        restored = np.fft.ifft2(restored_fft).real
        restored = np.clip(restored, 0, 1)
        
        return restored
    
    def wiener_filter(self, degraded_img, kernel, K=0.01):
        """Apply Wiener filtering"""
        # Pad kernel
        img_height, img_width = degraded_img.shape
        k_height, k_width = kernel.shape
        
        padded_kernel = np.zeros((img_height, img_width))
        start_h = (img_height - k_height) // 2
        start_w = (img_width - k_width) // 2
        padded_kernel[start_h:start_h+k_height, start_w:start_w+k_width] = kernel
        padded_kernel = np.fft.ifftshift(padded_kernel)
        
        # Compute FFTs
        degraded_fft = np.fft.fft2(degraded_img)
        kernel_fft = np.fft.fft2(padded_kernel)
        
        # Wiener filtering
        kernel_fft_conj = np.conjugate(kernel_fft)
        denominator = np.abs(kernel_fft)**2 + K
        wiener_filter = kernel_fft_conj / denominator
        
        restored_fft = degraded_fft * wiener_filter
        restored_fft = np.fft.ifftshift(restored_fft)
        restored = np.fft.ifft2(restored_fft).real
        restored = np.clip(restored, 0, 1)
        
        return restored
    
    def calculate_psnr(self, original, restored):
        """Calculate PSNR between original and restored images"""
        mse = np.mean((original - restored) ** 2)
        max_pixel = 1.0
        psnr = 10 * np.log10((max_pixel ** 2) / mse) if mse > 0 else 100
        return psnr
    
    def calculate_statistics(self, image):
        """Calculate mean and standard deviation"""
        mean = np.mean(image)
        std_dev = np.std(image)
        return mean, std_dev
    
    def plot_results(self, original, degraded, inverse_restored, wiener_restored, 
                    degraded_psnr, inverse_psnr, wiener_psnr):
        """Plot and save comparison results"""
        plt.figure(figsize=(12, 10))
        
        plt.subplot(2, 2, 1)
        plt.imshow(original, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(2, 2, 2)
        plt.imshow(degraded, cmap='gray')
        plt.title(f'Degraded Image (PSNR: {degraded_psnr:.2f} dB)')
        plt.axis('off')
        
        plt.subplot(2, 2, 3)
        plt.imshow(inverse_restored, cmap='gray')
        plt.title(f'Inverse Filter (PSNR: {inverse_psnr:.2f} dB)')
        plt.axis('off')
        
        plt.subplot(2, 2, 4)
        plt.imshow(wiener_restored, cmap='gray')
        plt.title(f'Wiener Filter (PSNR: {wiener_psnr:.2f} dB)')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'restoration_comparison.png'))
        plt.close()
    
    def run_experiment(self):
        """Run the complete image restoration experiment"""
        try:
            # Create Gaussian kernel
            kernel = self.create_gaussian_kernel(kernel_size=15, sigma=1.5)
            
            # Degrade image
            degraded, noise = self.degrade_image(kernel, noise_level=10E-4)
            
            # Apply restoration filters
            inverse_restored = self.inverse_filter(degraded, kernel, threshold=0.01)
            wiener_restored = self.wiener_filter(degraded, kernel, K=10E-6)
            
            # Calculate PSNR
            degraded_psnr = self.calculate_psnr(self.original, degraded)
            inverse_psnr = self.calculate_psnr(self.original, inverse_restored)
            wiener_psnr = self.calculate_psnr(self.original, wiener_restored)
            
            # Calculate statistics
            original_stats = self.calculate_statistics(self.original)
            degraded_stats = self.calculate_statistics(degraded)
            inverse_stats = self.calculate_statistics(inverse_restored)
            wiener_stats = self.calculate_statistics(wiener_restored)
            
            # Plot and save results
            self.plot_results(self.original, degraded, inverse_restored, wiener_restored,
                            degraded_psnr, inverse_psnr, wiener_psnr)
            
            # Save individual images
            plt.imsave(os.path.join(self.output_dir, 'original.png'), self.original, cmap='gray')
            plt.imsave(os.path.join(self.output_dir, 'degraded.png'), degraded, cmap='gray')
            plt.imsave(os.path.join(self.output_dir, 'inverse_restored.png'), inverse_restored, cmap='gray')
            plt.imsave(os.path.join(self.output_dir, 'wiener_restored.png'), wiener_restored, cmap='gray')
            
            # Print statistics
            print("\nFinal Statistics Summary:")
            print("=======================")
            print(f"{'Image Type':<25} {'Mean':<10} {'Std Dev':<10} {'PSNR (dB)':<10}")
            print("-" * 60)
            print(f"{'Original':<25} {original_stats[0]:<10.4f} {original_stats[1]:<10.4f} {'N/A':<10}")
            print(f"{'Degraded':<25} {degraded_stats[0]:<10.4f} {degraded_stats[1]:<10.4f} {degraded_psnr:<10.2f}")
            print(f"{'Inverse Filter Restored':<25} {inverse_stats[0]:<10.4f} {inverse_stats[1]:<10.4f} {inverse_psnr:<10.2f}")
            print(f"{'Wiener Filter Restored':<25} {wiener_stats[0]:<10.4f} {wiener_stats[1]:<10.4f} {wiener_psnr:<10.2f}")
            
        except Exception as e:
            print(f"Error: {str(e)}")

def main():
    try:
        restorer = ImageRestorer()
        restorer.run_experiment()
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 