#!/usr/bin/env python
"""
Script to download and set up the Shap-E model.

This script downloads the Shap-E model from the official repository
and sets it up for use in the RingGen pipeline.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_command(command, cwd=None):
    """Run a shell command and print output."""
    print(f"Running: {command}")
    result = subprocess.run(
        command, 
        shell=True, 
        cwd=cwd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT,
        text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}")
        return False
    return True


def install_shap_e():
    """Install Shap-E from the official repository."""
    print("\n=== Installing Shap-E ===\n")
    
    # Check if Shap-E is already installed
    try:
        import shap_e
        print("Shap-E is already installed.")
        return True
    except ImportError:
        pass
    
    # Install Shap-E from the official repository
    command = "pip install git+https://github.com/openai/shap-e.git"
    return run_command(command)


def download_shap_e_models(output_dir):
    """Download Shap-E models."""
    print("\n=== Downloading Shap-E Models ===\n")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a simple script to download the models
    script_path = os.path.join(output_dir, "download_models.py")
    with open(script_path, "w") as f:
        f.write("""
import os
import torch
from shap_e.models.download import load_model, load_config

# Set the output directory
output_dir = os.path.dirname(os.path.abspath(__file__))

# Download the transmitter model
print("Downloading transmitter model...")
transmitter = load_model('transmitter', device='cpu')
torch.save(transmitter, os.path.join(output_dir, 'transmitter.pt'))

# Download the text-to-3d model
print("Downloading text-to-3d model...")
text_to_3d = load_model('text300M', device='cpu')
torch.save(text_to_3d, os.path.join(output_dir, 'text300M.pt'))

# Download the diffusion config
print("Downloading diffusion config...")
diffusion_config = load_config('diffusion')
torch.save(diffusion_config, os.path.join(output_dir, 'diffusion_config.pt'))

print("All models downloaded successfully!")
""")
    
    # Run the script
    command = f"python {script_path}"
    return run_command(command)


def update_config_for_shap_e(config_dir, models_dir):
    """Update configuration files to use the downloaded Shap-E models."""
    print("\n=== Updating Configuration for Shap-E ===\n")
    
    # Update generation config
    generation_config_path = os.path.join(config_dir, "generation.yaml")
    if os.path.exists(generation_config_path):
        with open(generation_config_path, "r") as f:
            content = f.read()
        
        # Update model paths
        content = content.replace(
            "models:\n  prior: \"outputs/prior_checkpoints/best_model.pt\"\n  decoder: \"outputs/decoder_checkpoints/best_model.pt\"",
            f"models:\n  prior: \"outputs/prior_checkpoints/best_model.pt\"\n  decoder: \"outputs/decoder_checkpoints/best_model.pt\"\n  shap_e_transmitter: \"{os.path.join(models_dir, 'transmitter.pt')}\"\n  shap_e_text: \"{os.path.join(models_dir, 'text300M.pt')}\"\n  shap_e_diffusion: \"{os.path.join(models_dir, 'diffusion_config.pt')}\""
        )
        
        with open(generation_config_path, "w") as f:
            f.write(content)
        
        print(f"Updated {generation_config_path}")
    
    # Update encoding module to use the downloaded models
    encoding_path = os.path.join("ringgen", "encoding", "shap_e.py")
    if os.path.exists(encoding_path):
        with open(encoding_path, "r") as f:
            content = f.read()
        
        # Update model loading
        content = content.replace(
            "        if model_path and os.path.exists(model_path):\n            # Load from local path\n            self.xm = torch.load(model_path, map_location=self.device)\n        else:\n            # Load from pretrained\n            self.xm = load_model('transmitter', device=self.device)",
            f"        if model_path and os.path.exists(model_path):\n            # Load from local path\n            self.xm = torch.load(model_path, map_location=self.device)\n        else:\n            # Try to load from downloaded models\n            transmitter_path = os.path.join('{models_dir}', 'transmitter.pt')\n            if os.path.exists(transmitter_path):\n                self.xm = torch.load(transmitter_path, map_location=self.device)\n            else:\n                # Load from pretrained\n                self.xm = load_model('transmitter', device=self.device)"
        )
        
        with open(encoding_path, "w") as f:
            f.write(content)
        
        print(f"Updated {encoding_path}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Download and set up Shap-E model")
    parser.add_argument("--output-dir", type=str, default="models/shap_e", 
                        help="Directory to store Shap-E models")
    parser.add_argument("--config-dir", type=str, default="configs", 
                        help="Directory containing configuration files")
    parser.add_argument("--skip-install", action="store_true", 
                        help="Skip installing Shap-E")
    parser.add_argument("--skip-download", action="store_true", 
                        help="Skip downloading Shap-E models")
    parser.add_argument("--skip-config-update", action="store_true", 
                        help="Skip updating configuration files")
    args = parser.parse_args()
    
    # Install Shap-E
    if not args.skip_install:
        if not install_shap_e():
            print("Failed to install Shap-E!")
            return 1
    
    # Download Shap-E models
    if not args.skip_download:
        if not download_shap_e_models(args.output_dir):
            print("Failed to download Shap-E models!")
            return 1
    
    # Update configuration
    if not args.skip_config_update:
        if not update_config_for_shap_e(args.config_dir, args.output_dir):
            print("Failed to update configuration!")
            return 1
    
    print("\n=== Shap-E Setup Completed Successfully ===\n")
    print(f"Shap-E models are stored in: {args.output_dir}")
    print("You can now use Shap-E in the RingGen pipeline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
