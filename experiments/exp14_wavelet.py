import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt
import os

class WaveletDecomposer:
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
    
    def decompose(self, wavelet: str = 'haar', level: int = 1):
        """
        Perform wavelet decomposition.
        
        Args:
            wavelet (str): Wavelet type ('haar', 'db1', etc.)
            level (int): Decomposition level
            
        Returns:
            tuple: Coefficients (cA, (cH, cV, cD))
        """
        return pywt.wavedec2(self.original, wavelet, level=level)
    
    def visualize_coefficients(self, coeffs, titles=None):
        """
        Visualize wavelet coefficients.
        
        Args:
            coeffs: Wavelet coefficients from wavedec2
            titles: Optional list of titles for subplots
        """
        # Get coefficients
        cA, (cH, cV, cD) = coeffs
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        
        # Plot approximation and details
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
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'decomposition.png'))
        plt.close()
    
    def analyze_energy_distribution(self, coeffs):
        """
        Analyze energy distribution in wavelet coefficients.
        
        Args:
            coeffs: Wavelet coefficients from wavedec2
        """
        cA, (cH, cV, cD) = coeffs
        
        # Calculate energy in each component
        total_energy = (np.sum(cA**2) + np.sum(cH**2) + 
                       np.sum(cV**2) + np.sum(cD**2))
        
        energies = {
            'Approximation (LL)': np.sum(cA**2) / total_energy * 100,
            'Horizontal (LH)': np.sum(cH**2) / total_energy * 100,
            'Vertical (HL)': np.sum(cV**2) / total_energy * 100,
            'Diagonal (HH)': np.sum(cD**2) / total_energy * 100
        }
        
        # Plot energy distribution
        plt.figure(figsize=(10, 5))
        plt.bar(energies.keys(), energies.values())
        plt.title('Energy Distribution in Wavelet Components')
        plt.ylabel('Energy (%)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'energy_distribution.png'))
        plt.close()
        
        return energies

def main():
    try:
        # Initialize decomposer
        decomposer = WaveletDecomposer()
        
        # Perform wavelet decomposition
        coeffs = decomposer.decompose(wavelet='haar', level=1)
        
        # Visualize coefficients
        decomposer.visualize_coefficients(coeffs)
        
        # Analyze energy distribution
        energies = decomposer.analyze_energy_distribution(coeffs)
        
        # Print energy distribution
        print("\nEnergy Distribution:")
        for component, energy in energies.items():
            print(f"{component}: {energy:.2f}%")
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 