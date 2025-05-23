import os
import torch
import sys
from shap_e.models.download import load_model, load_config

def main():
    # Set the output directory
    output_dir = os.path.dirname(os.path.abspath(__file__))
    
    try:
        # Download the text-to-3d model
        print("Downloading text-to-3d model...")
        text_to_3d = load_model('text300M', device='cpu')
        
        # Download the diffusion config
        print("Downloading diffusion config...")
        diffusion_config = load_config('diffusion')
        torch.save(diffusion_config, os.path.join(output_dir, 'diffusion_config.pt'))
        
        print("Models loaded successfully!")
        
        # Instead of saving the transmitter model directly, we'll use it to generate a dummy output
        # This ensures the model is downloaded and cached by Shap-E
        print("Caching transmitter model...")
        transmitter = load_model('transmitter', device='cpu')
        print("Transmitter model cached successfully!")
        
        print("All models downloaded and cached successfully!")
        return 0
    except Exception as e:
        print(f"Error downloading models: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
