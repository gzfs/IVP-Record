import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dct, idct
from typing import Tuple, List, Optional
import os

class SequenceEnergyAnalyzer:
    def __init__(self):
        """
        Initialize the sequence energy analyzer.
        """
        self.sequence = None
        # Create output directory
        self.output_dir = "../output/sequence_energy"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def generate_test_sequence(self, length: int = 128,
                             noise_std: float = 0.1) -> np.ndarray:
        """
        Generate a test sequence with multiple components and noise.
        
        Args:
            length (int): Length of the sequence
            noise_std (float): Standard deviation of noise
            
        Returns:
            numpy.ndarray: Generated sequence
        """
        t = np.linspace(0, 1, length)
        
        # Generate clean signal
        signal = (np.sin(2 * np.pi * 2 * t) +  # Low frequency
                 0.5 * np.sin(2 * np.pi * 10 * t) +  # Medium frequency
                 0.25 * np.sin(2 * np.pi * 20 * t))  # High frequency
        
        # Add noise
        noise = np.random.normal(0, noise_std, length)
        sequence = signal + noise
        
        self.sequence = sequence
        return sequence
    
    def compute_energy(self, signal: np.ndarray) -> float:
        """
        Compute the energy of a signal.
        
        Args:
            signal (numpy.ndarray): Input signal
            
        Returns:
            float: Signal energy
        """
        return np.sum(np.abs(signal) ** 2)
    
    def save_plot(self, filename, plot_func):
        """Helper function to save plots"""
        plt.figure(figsize=(10, 4))
        plot_func()
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
    
    def analyze_domain_energy(self):
        """Analyze energy in time and frequency domains"""
        if self.sequence is None:
            raise ValueError("No sequence available for analysis")
            
        # Compute DCT
        dct_coeffs = dct(self.sequence, type=2, norm='ortho')
        
        # Compute energies
        time_energy = self.compute_energy(self.sequence)
        freq_energy = self.compute_energy(dct_coeffs)
        
        # Save time domain plot
        def plot_time_domain():
            plt.plot(self.sequence)
            plt.title(f'Time Domain\nEnergy: {time_energy:.2f}')
            plt.xlabel('Sample Index')
            plt.ylabel('Amplitude')
            plt.grid(True)
        
        self.save_plot('time_domain.png', plot_time_domain)
        
        # Save frequency domain plot
        def plot_freq_domain():
            plt.stem(dct_coeffs)
            plt.title(f'DCT Domain\nEnergy: {freq_energy:.2f}')
            plt.xlabel('Coefficient Index')
            plt.ylabel('Magnitude')
            plt.grid(True)
        
        self.save_plot('frequency_domain.png', plot_freq_domain)
        
        # Save energy data
        np.save(os.path.join(self.output_dir, 'domain_energy.npy'), {
            'time_energy': time_energy,
            'freq_energy': freq_energy
        })
        
        return time_energy, freq_energy
    
    def analyze_energy_compaction(self, thresholds: List[float] = [0.8, 0.9, 0.95, 0.99]):
        """
        Analyze energy compaction in DCT domain.
        
        Args:
            thresholds (list): Energy retention thresholds to analyze
        """
        if self.sequence is None:
            raise ValueError("No sequence available for analysis")
            
        # Compute DCT
        dct_coeffs = dct(self.sequence, type=2, norm='ortho')
        
        # Sort coefficients by magnitude
        sorted_coeffs = np.sort(np.abs(dct_coeffs))[::-1]
        
        # Calculate cumulative energy
        energy = sorted_coeffs ** 2
        total_energy = np.sum(energy)
        cumulative_energy = np.cumsum(energy) / total_energy
        
        # Save sorted coefficients plot
        def plot_sorted_coeffs():
            plt.plot(sorted_coeffs, 'b-')
            plt.title('Sorted DCT Coefficients')
            plt.xlabel('Index')
            plt.ylabel('Magnitude')
            plt.grid(True)
        
        self.save_plot('sorted_coefficients.png', plot_sorted_coeffs)
        
        # Save energy distribution plot
        def plot_energy_dist():
            plt.plot(cumulative_energy, 'r-')
            plt.title('Cumulative Energy Distribution')
            plt.xlabel('Number of Coefficients')
            plt.ylabel('Energy Ratio')
            plt.grid(True)
            
            # Add threshold lines
            for threshold in thresholds:
                n_coeffs = np.where(cumulative_energy >= threshold)[0][0] + 1
                plt.axhline(y=threshold, color='g', linestyle='--', alpha=0.5)
                plt.axvline(x=n_coeffs, color='g', linestyle='--', alpha=0.5)
        
        self.save_plot('energy_distribution.png', plot_energy_dist)
        
        # Save analysis results
        analysis = {}
        for threshold in thresholds:
            n_coeffs = np.where(cumulative_energy >= threshold)[0][0] + 1
            analysis[threshold] = {
                'n_coeffs': n_coeffs,
                'ratio': n_coeffs/len(dct_coeffs)
            }
        np.save(os.path.join(self.output_dir, 'compaction_analysis.npy'), analysis)
    
    def analyze_reconstruction_quality(self, keep_ratios: List[float] = [1.0, 0.5, 0.25, 0.1]):
        """
        Analyze reconstruction quality with different numbers of coefficients.
        
        Args:
            keep_ratios (list): Ratios of coefficients to keep
        """
        if self.sequence is None:
            raise ValueError("No sequence available for analysis")
            
        # Compute DCT
        dct_coeffs = dct(self.sequence, type=2, norm='ortho')
        
        # Analyze each ratio
        for ratio in keep_ratios:
            # Keep only specified ratio of coefficients
            n_keep = int(len(dct_coeffs) * ratio)
            modified_coeffs = dct_coeffs.copy()
            modified_coeffs[n_keep:] = 0
            
            # Reconstruct signal
            reconstructed = idct(modified_coeffs, type=2, norm='ortho')
            
            # Calculate error metrics
            mse = np.mean((self.sequence - reconstructed) ** 2)
            max_error = np.max(np.abs(self.sequence - reconstructed))
            retained_energy = np.sum(modified_coeffs**2) / np.sum(dct_coeffs**2)
            
            # Save reconstruction plot
            def plot_reconstruction():
                plt.plot(self.sequence, 'b-', label='Original')
                plt.plot(reconstructed, 'r--', label='Reconstructed')
                plt.title(f'Reconstruction with {ratio:.0%} coefficients\n'
                         f'MSE: {mse:.6f}, Max Error: {max_error:.6f}, '
                         f'Energy Retained: {retained_energy:.1%}')
                plt.grid(True)
                plt.legend()
            
            self.save_plot(f'reconstruction_{int(ratio*100)}percent.png', plot_reconstruction)
            
            # Save metrics
            np.save(os.path.join(self.output_dir, f'metrics_{int(ratio*100)}percent.npy'), {
                'ratio': ratio,
                'mse': mse,
                'max_error': max_error,
                'retained_energy': retained_energy
            })

def main():
    try:
        # Create analyzer and generate test sequence
        analyzer = SequenceEnergyAnalyzer()
        sequence = analyzer.generate_test_sequence(length=128, noise_std=0.1)
        
        # Save original sequence
        def plot_sequence():
            plt.plot(sequence)
            plt.title('Original Test Sequence')
            plt.grid(True)
        
        analyzer.save_plot('sequence_energy.png', plot_sequence)
        
        # Run all analyses
        analyzer.analyze_domain_energy()
        analyzer.analyze_energy_compaction([0.8, 0.9, 0.95, 0.99])
        analyzer.analyze_reconstruction_quality([1.0, 0.5, 0.25, 0.1])
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()