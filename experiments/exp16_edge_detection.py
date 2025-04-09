import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

class EdgeDetector:
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
        self.output_dir = "../output/edge_detection"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def basic_edge_detection(self):
        """Apply basic edge detection methods (Laplacian and Sobel)"""
        # Convert to float64 for better precision
        img = np.float64(self.original)
        
        # Apply edge detection
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        
        # Create figure
        plt.figure(figsize=(12, 12))
        
        # Plot results
        plt.subplot(2, 2, 1)
        plt.imshow(self.original, cmap='gray')
        plt.title('Original')
        plt.xticks([]), plt.yticks([])
        
        plt.subplot(2, 2, 2)
        plt.imshow(laplacian, cmap='gray')
        plt.title('Laplacian')
        plt.xticks([]), plt.yticks([])
        
        plt.subplot(2, 2, 3)
        plt.imshow(sobelx, cmap='gray')
        plt.title('Sobel X')
        plt.xticks([]), plt.yticks([])
        
        plt.subplot(2, 2, 4)
        plt.imshow(sobely, cmap='gray')
        plt.title('Sobel Y')
        plt.xticks([]), plt.yticks([])
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'basic_edge_detection.png'))
        plt.close()
    
    def canny_edge_detection(self):
        """Apply Canny edge detection"""
        canny_edges = cv2.Canny(self.original, 100, 200)
        
        plt.figure(figsize=(8, 8))
        plt.imshow(canny_edges, cmap='gray')
        plt.title('Canny Edge Detection')
        plt.xticks([]), plt.yticks([])
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'canny_edges.png'))
        plt.close()
    
    def gaussian_kernel(self, size, sigma):
        """Generate 1D Gaussian kernel"""
        kernel = np.zeros(size)
        center = size // 2
        for i in range(size):
            kernel[i] = np.exp(-(i - center) ** 2 / (2 * sigma ** 2))
        return kernel / np.sum(kernel)
    
    def dog_kernel(self, size, sigma):
        """Generate 1D Derivative of Gaussian kernel"""
        g = self.gaussian_kernel(size, sigma)
        dg = np.zeros_like(g)
        for i in range(1, size - 1):
            dg[i] = (g[i + 1] - g[i - 1]) / 2
        return dg / np.sum(np.abs(dg))
    
    def apply_dog_2d(self, image, kernel_x, kernel_y):
        """Apply 2D Derivative of Gaussian"""
        grad_x = cv2.filter2D(image, -1, kernel_x.reshape(1, -1))
        grad_y = cv2.filter2D(image, -1, kernel_y.reshape(-1, 1))
        return grad_x, grad_y
    
    def derivative_of_gaussian(self):
        """Apply Derivative of Gaussian edge detection"""
        # Convert to float64
        image = np.float64(self.original)
        
        # Parameters
        kernel_size = 15
        sigma = 2.0
        
        # Generate kernels
        kernel_x = self.dog_kernel(kernel_size, sigma)
        kernel_y = self.dog_kernel(kernel_size, sigma)
        
        # Apply DoG
        grad_x, grad_y = self.apply_dog_2d(image, kernel_x, kernel_y)
        gradient_magnitude = cv2.magnitude(grad_x, grad_y)
        
        # Create figure
        plt.figure(figsize=(12, 12))
        
        # Plot results
        plt.subplot(3, 3, 1)
        plt.imshow(image, cmap='gray')
        plt.title('Original Image')
        plt.xticks([]), plt.yticks([])
        
        plt.subplot(3, 3, 2)
        plt.plot(kernel_x, label='1D DoG - X Kernel')
        plt.title('1D Derivative of Gaussian (X-axis)')
        plt.legend()
        
        plt.subplot(3, 3, 3)
        plt.plot(kernel_y, label='1D DoG - Y Kernel')
        plt.title('1D Derivative of Gaussian (Y-axis)')
        plt.legend()
        
        plt.subplot(3, 3, 4)
        plt.imshow(grad_x, cmap='gray')
        plt.title('Gradient X')
        plt.xticks([]), plt.yticks([])
        
        plt.subplot(3, 3, 5)
        plt.imshow(grad_y, cmap='gray')
        plt.title('Gradient Y')
        plt.xticks([]), plt.yticks([])
        
        plt.subplot(3, 3, 6)
        plt.imshow(gradient_magnitude, cmap='gray')
        plt.title('Derivative Of Gaussian')
        plt.xticks([]), plt.yticks([])
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'derivative_of_gaussian.png'))
        plt.close()
    
    def log_kernel(self, size, sigma):
        """Generate 2D Laplacian of Gaussian kernel"""
        x = np.linspace(-size // 2, size // 2, size)
        y = np.linspace(-size // 2, size // 2, size)
        X, Y = np.meshgrid(x, y)
        
        gaussian_2d = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
        log_2d = -(1 / (np.pi * sigma**4)) * (1 - (X**2 + Y**2) / (2 * sigma**2)) * gaussian_2d
        
        return log_2d
    
    def laplacian_of_gaussian(self):
        """Apply Laplacian of Gaussian edge detection"""
        # Convert to float64
        image = np.float64(self.original)
        
        # Parameters
        kernel_size = 15
        sigma = 2.0
        
        # Generate and apply LoG kernel
        log_kernel_2d = self.log_kernel(kernel_size, sigma)
        log_result = cv2.filter2D(image, -1, log_kernel_2d)
        
        # Create figure
        plt.figure(figsize=(12, 12))
        
        # Plot results
        plt.subplot(1, 2, 1)
        plt.imshow(image, cmap='gray')
        plt.title('Original Image')
        plt.xticks([]), plt.yticks([])
        
        plt.subplot(1, 2, 2)
        plt.imshow(log_result, cmap='gray')
        plt.title('LoG Image Result')
        plt.xticks([]), plt.yticks([])
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'laplacian_of_gaussian.png'))
        plt.close()

def main():
    try:
        # Initialize detector
        detector = EdgeDetector()
        
        # Run all edge detection methods
        detector.basic_edge_detection()
        detector.canny_edge_detection()
        detector.derivative_of_gaussian()
        detector.laplacian_of_gaussian()
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 