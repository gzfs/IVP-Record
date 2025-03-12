import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dct, idct
from typing import Tuple, List, Optional
import os

class DCTAnalyzer:
    def __init__(self):
        """
        Initialize the DCT analyzer.
        """
        self.sequence = None
        # Create output directory
        self.output_dir = "../output/dct"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def generate_test_sequence(self, length: int = 64, 
                             frequencies: List[float] = [1, 5, 10],
                             amplitudes: List[float] = [1, 0.5, 0.25]) -> np.ndarray:
        """
        Generate a test sequence with multiple frequency components.
        
        Args:
            length (int): Length of the sequence
            frequencies (list): List of frequencies to include
            amplitudes (list): List of corresponding amplitudes
            
        Returns:
            numpy.ndarray: Generated sequence
        """
        t = np.linspace(0, 1, length)
        sequence = np.zeros_like(t)
        
        for freq, amp in zip(frequencies, amplitudes):
            sequence += amp * np.sin(2 * np.pi * freq * t)
        
        self.sequence = sequence
        return sequence
    
    def compute_dct(self, sequence: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute the DCT of a sequence.
        
        Args:
            sequence (numpy.ndarray, optional): Input sequence
            
        Returns:
            numpy.ndarray: DCT coefficients
        """
        if sequence is None:
            sequence = self.sequence
        if sequence is None:
            raise ValueError("No sequence provided")
            
        return dct(sequence, type=2, norm='ortho')
    
    def compute_idct(self, coefficients: np.ndarray) -> np.ndarray:
        """
        Compute the inverse DCT.
        
        Args:
            coefficients (numpy.ndarray): DCT coefficients
            
        Returns:
            numpy.ndarray: Reconstructed sequence
        """
        return idct(coefficients, type=2, norm='ortho')
    
    def modify_coefficients(self, coefficients: np.ndarray, 
                          keep_ratio: float = 0.5) -> np.ndarray:
        """
        Modify DCT coefficients by keeping only a portion of them.
        
        Args:
            coefficients (numpy.ndarray): Original DCT coefficients
            keep_ratio (float): Ratio of coefficients to keep
            
        Returns:
            numpy.ndarray: Modified coefficients
        """
        n_coeffs = len(coefficients)
        n_keep = int(n_coeffs * keep_ratio)
        
        modified = coefficients.copy()
        modified[n_keep:] = 0
        
        return modified
    
    def save_plot(self, filename, plot_func):
        """Helper function to save plots"""
        plt.figure(figsize=(10, 4))
        plot_func()
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
    
    def analyze_compression(self, keep_ratios: List[float] = [1.0, 0.5, 0.25, 0.1]):
        """
        Analyze compression effects with different ratios of kept coefficients.
        
        Args:
            keep_ratios (list): List of ratios of coefficients to keep
        """
        if self.sequence is None:
            raise ValueError("No sequence available for analysis")
            
        # Compute DCT
        coefficients = self.compute_dct()
        
        for ratio in keep_ratios:
            # Modify coefficients and reconstruct
            modified_coeffs = self.modify_coefficients(coefficients, ratio)
            reconstructed = self.compute_idct(modified_coeffs)
            
            # Save reconstruction plot
            def plot_reconstruction():
                plt.plot(self.sequence, 'b-', label='Original')
                plt.plot(reconstructed, 'r--', label='Reconstructed')
                plt.title(f'Reconstruction with {ratio:.0%} coefficients')
                plt.grid(True)
                plt.legend()
            
            self.save_plot(f'reconstruction_{int(ratio*100)}percent.png', plot_reconstruction)
            
            # Calculate and save metrics
            mse = np.mean((self.sequence - reconstructed) ** 2)
            max_error = np.max(np.abs(self.sequence - reconstructed))
            energy_retention = np.sum(modified_coeffs**2)/np.sum(coefficients**2)
            
            metrics = {
                'ratio': ratio,
                'mse': mse,
                'max_error': max_error,
                'energy_retention': energy_retention
            }
            np.save(os.path.join(self.output_dir, f'metrics_{int(ratio*100)}percent.npy'), metrics)
    
    def analyze_energy_compaction(self):
        """Analyze energy compaction property of DCT"""
        if self.sequence is None:
            raise ValueError("No sequence available for analysis")
            
        # Compute DCT
        coefficients = self.compute_dct()
        
        # Calculate cumulative energy
        energy = coefficients ** 2
        total_energy = np.sum(energy)
        cumulative_energy = np.cumsum(energy) / total_energy
        
        # Save coefficients plot
        def plot_coefficients():
            plt.stem(coefficients)
            plt.title('DCT Coefficients')
            plt.xlabel('Coefficient Index')
            plt.ylabel('Magnitude')
            plt.grid(True)
        
        self.save_plot('dct_coefficients.png', plot_coefficients)
        
        # Save energy distribution plot
        def plot_energy():
            plt.plot(cumulative_energy)
            plt.title('Cumulative Energy Distribution')
            plt.xlabel('Number of Coefficients')
            plt.ylabel('Cumulative Energy Ratio')
            plt.grid(True)
        
        self.save_plot('energy_distribution.png', plot_energy)
        
        # Save energy analysis data
        analysis = {}
        for threshold in [0.8, 0.9, 0.95, 0.99]:
            n_coeffs = np.where(cumulative_energy >= threshold)[0][0] + 1
            analysis[threshold] = {
                'n_coeffs': n_coeffs,
                'ratio': n_coeffs/len(coefficients)
            }
        np.save(os.path.join(self.output_dir, 'energy_analysis.npy'), analysis)

def main():
    try:
        # Create analyzer and generate test sequence
        analyzer = DCTAnalyzer()
        sequence = analyzer.generate_test_sequence(
            length=128,
            frequencies=[1, 5, 10, 20],
            amplitudes=[1, 0.5, 0.25, 0.1]
        )
        
        # Save original sequence plot
        def plot_original():
            plt.plot(sequence)
            plt.title('Original Test Sequence')
            plt.grid(True)
        
        analyzer.save_plot('original_sequence.png', plot_original)
        
        # Analyze compression with different ratios
        keep_ratios = [1.0, 0.5, 0.25, 0.1]
        analyzer.analyze_compression(keep_ratios)
        
        # Analyze energy compaction
        analyzer.analyze_energy_compaction()
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 