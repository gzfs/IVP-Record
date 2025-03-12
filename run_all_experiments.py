import os
import sys
import importlib.util
import shutil

def import_experiment(filename):
    """Import an experiment module dynamically."""
    module_name = filename[:-3]  # Remove .py extension
    spec = importlib.util.spec_from_file_location(module_name, filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def run_all_experiments():
    """Run all experiment scripts."""
    # Create output directory
    os.makedirs("output", exist_ok=True)
    
    # List of experiment files to run
    experiment_files = [
        "exp1_eye_detection.py",
        "exp2_image_transformations.py",
        "exp3_rgb_to_gray.py",
        "exp4_intensity_resolution.py",
        "exp5_spatial_resolution.py",
        "exp6_interpolation.py",
        "exp7_image_blurring.py",
        "exp8_histogram_processing.py",
        "exp9_freq_domain_box.py",
        "exp10_freq_domain_gaussian.py",
        "exp11_freq_domain_random.py",
        "exp12_dct_sequence.py",
        "exp13_sequence_energy.py",
        "exp14_wavelet.py",
        "exp15_wavelet_denoising.py"
    ]
    
    # Change to experiments directory to run the scripts
    original_dir = os.getcwd()
    os.chdir("experiments")
    
    # Copy images to current directory if they don't exist
    if not os.path.exists("image.png") and os.path.exists("../experiments/image.png"):
        print("Copying image.png to current directory")
        shutil.copy("../experiments/image.png", "image.png")
    
    if not os.path.exists("image_two.png") and os.path.exists("../experiments/image_two.png"):
        print("Copying image_two.png to current directory")
        shutil.copy("../experiments/image_two.png", "image_two.png")
    
    # Run each experiment
    for filename in experiment_files:
        try:
            print(f"\nRunning {filename}...")
            if os.path.exists(filename):
                module = import_experiment(filename)
                if hasattr(module, 'main'):
                    module.main()
                print(f"Completed {filename}")
            else:
                print(f"Warning: {filename} not found")
        except Exception as e:
            print(f"Error in {filename}: {str(e)}")
    
    # Change back to original directory
    os.chdir(original_dir)

if __name__ == "__main__":
    run_all_experiments() 