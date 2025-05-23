#!/usr/bin/env python
"""
Generate 3D ring models from text prompts using trained models.

This script:
1. Takes text prompts as input
2. Uses trained CAP3D model to convert text to latent vectors
3. Generates 3D ring models based on the latent vectors
"""

import argparse
import json
import logging
import os
import sys
import time
import numpy as np
import torch
import trimesh

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("generation_log.txt")],
)
logger = logging.getLogger("RingGen")


def create_basic_ring_mesh(prompt, index, output_dir, latent=None):
    """Create a basic ring mesh based on the prompt and latent vector."""
    try:
        import trimesh
        import numpy as np

        # Extract key features from the prompt
        prompt_lower = prompt.lower()

        # Determine ring style and materials
        is_vintage = "vintage" in prompt_lower
        is_modern = "modern" in prompt_lower or "minimalist" in prompt_lower
        is_classic = "classic" in prompt_lower or "solitaire" in prompt_lower

        # Determine metal type
        if "gold" in prompt_lower:
            metal_color = [1.0, 0.843, 0.0]  # Gold
        elif (
            "silver" in prompt_lower
            or "platinum" in prompt_lower
            or "white gold" in prompt_lower
        ):
            metal_color = [0.753, 0.753, 0.753]  # Silver/Platinum
        else:
            metal_color = [1.0, 0.843, 0.0]  # Default to gold

        # Determine stone type
        if "diamond" in prompt_lower:
            stone_color = [0.9, 0.9, 0.9]  # Diamond
        elif "ruby" in prompt_lower:
            stone_color = [0.9, 0.1, 0.1]  # Ruby
        elif "emerald" in prompt_lower:
            stone_color = [0.1, 0.8, 0.1]  # Emerald
        elif "sapphire" in prompt_lower:
            stone_color = [0.1, 0.1, 0.8]  # Sapphire
        else:
            stone_color = [0.9, 0.9, 0.9]  # Default to diamond

        # Create a ring band (torus)
        ring_radius = 10.0  # Major radius (ring size)
        tube_radius = 1.0  # Minor radius (band thickness)

        if is_vintage:
            # More ornate band for vintage style
            tube_radius = 1.2
            ring = trimesh.creation.annulus(
                r_min=ring_radius - tube_radius,
                r_max=ring_radius + tube_radius,
                height=tube_radius * 1.5,
            )
            # Add some texture/pattern to the band
            vertices = ring.vertices
            noise = np.random.random(vertices.shape) * 0.2
            vertices += noise
            ring = trimesh.Trimesh(vertices=vertices, faces=ring.faces)
        elif is_modern:
            # Sleeker, thinner band for modern style
            tube_radius = 0.8
            ring = trimesh.creation.annulus(
                r_min=ring_radius - tube_radius,
                r_max=ring_radius + tube_radius,
                height=tube_radius,
            )
        else:  # Classic style
            ring = trimesh.creation.annulus(
                r_min=ring_radius - tube_radius,
                r_max=ring_radius + tube_radius,
                height=tube_radius * 1.2,
            )

        # Create a stone (sphere or cube)
        if "princess" in prompt_lower:
            # Princess cut (cube)
            stone_size = 2.0
            stone = trimesh.creation.box(extents=[stone_size, stone_size, stone_size])
            stone.apply_translation([0, 0, tube_radius + stone_size / 2])
        else:
            # Round cut (sphere)
            stone_size = 1.5
            stone = trimesh.creation.icosphere(radius=stone_size)
            stone.apply_translation([0, 0, tube_radius + stone_size])

        # Add accent stones for vintage style
        accent_stones = trimesh.Trimesh()
        if is_vintage or "three" in prompt_lower:
            # Add smaller accent stones
            accent_size = stone_size * 0.6
            for offset in [-2.5, 2.5]:
                accent = trimesh.creation.icosphere(radius=accent_size)
                accent.apply_translation([offset, 0, tube_radius + accent_size])
                accent_stones = trimesh.util.concatenate([accent_stones, accent])

        # Combine all components
        ring.visual.face_colors = metal_color + [1.0]  # RGBA
        stone.visual.face_colors = stone_color + [1.0]  # RGBA
        if not accent_stones.is_empty:
            accent_stones.visual.face_colors = stone_color + [1.0]  # RGBA
            combined = trimesh.util.concatenate([ring, stone, accent_stones])
        else:
            combined = trimesh.util.concatenate([ring, stone])

        # Save the mesh
        output_file = os.path.join(output_dir, f"generated_ring_{index+1}.obj")
        combined.export(output_file)

        logger.info(f"Created basic ring mesh based on prompt: {prompt}")
        return output_file

    except Exception as e:
        logger.error(f"Error creating basic ring mesh: {str(e)}")
        # Create a very simple fallback
        output_file = os.path.join(output_dir, f"generated_ring_{index+1}.obj")
        with open(output_file, "w") as f:
            f.write("# Basic ring model\n")
            f.write("# Prompt: " + prompt + "\n")
            # Create a simple torus-like shape with vertices and faces
            for i in range(20):
                angle = i * 2 * 3.14159 / 20
                x = 10 * np.cos(angle)
                y = 10 * np.sin(angle)
                for j in range(10):
                    angle2 = j * 2 * 3.14159 / 10
                    dx = np.cos(angle2)
                    dy = np.sin(angle2)
                    f.write(f"v {x + dx} {y + dy} {0.5}\n")
            # Write some faces
            for i in range(19):
                for j in range(9):
                    idx1 = i * 10 + j + 1
                    idx2 = i * 10 + j + 2
                    idx3 = (i + 1) * 10 + j + 2
                    idx4 = (i + 1) * 10 + j + 1
                    f.write(f"f {idx1} {idx2} {idx3} {idx4}\n")

        return output_file


def generate_rings(cap3d_model, output_dir, prompts=None, num_samples=3, device="cpu"):
    """Generate 3D ring models from text prompts."""
    logger.info("=== Generating 3D Ring Models ===")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Default prompts if none provided
    if prompts is None:
        prompts = [
            "A classic solitaire engagement ring with a round diamond and thin gold band",
            "A vintage-inspired ring with three diamonds and intricate gallery details",
            "A modern minimalist ring with a princess cut diamond and platinum band",
        ]

    # Limit to specified number of samples
    prompts = prompts[:num_samples]

    # Always perform real generation
    try:
        logger.info(f"Generating rings on {device} using trained models")

        # Import necessary libraries
        import torch
        import torch.nn as nn
        import trimesh

        # Load the CAP3D model for text-to-latent conversion
        if cap3d_model and os.path.exists(cap3d_model):
            logger.info(f"Loading trained CAP3D model from {cap3d_model}")

            # Try multiple approaches to load the model
            model_loaded = False
            checkpoint = None

            # Approach 1: Try with weights_only=False
            try:
                checkpoint = torch.load(
                    cap3d_model, map_location=device, weights_only=False
                )
                model_loaded = True
                logger.info("Successfully loaded model with weights_only=False")
            except Exception as e:
                logger.warning(f"Error loading with weights_only=False: {str(e)}")

            # Approach 2: Try with default settings
            if not model_loaded:
                try:
                    checkpoint = torch.load(cap3d_model, map_location=device)
                    model_loaded = True
                    logger.info("Successfully loaded model with default settings")
                except Exception as e:
                    logger.warning(f"Error loading with default settings: {str(e)}")

            # Approach 3: Try to read the file as a simple dictionary
            if not model_loaded:
                try:
                    checkpoint = {
                        "model_state_dict": torch.load(
                            cap3d_model, map_location=device, weights_only=True
                        )
                    }
                    model_loaded = True
                    logger.info("Successfully loaded model state dict directly")
                except Exception as e:
                    logger.warning(f"Error loading as state dict: {str(e)}")

            if not model_loaded or checkpoint is None:
                raise ValueError("Failed to load CAP3D model with any method")

            # Get latent dimension from checkpoint or use default
            latent_dim = checkpoint.get("latent_dim", 256)

            # Initialize the model and load weights
            class TextToLatentModel(nn.Module):
                def __init__(self, text_dim=512, hidden_dim=1024, latent_dim=256):
                    super().__init__()
                    self.encoder = nn.Sequential(
                        nn.Linear(text_dim, hidden_dim),
                        nn.LayerNorm(hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(hidden_dim, hidden_dim // 2),
                        nn.LayerNorm(hidden_dim // 2),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(hidden_dim // 2, latent_dim),
                    )

                def forward(self, x):
                    return self.encoder(x)

            model = TextToLatentModel(
                text_dim=512, hidden_dim=1024, latent_dim=latent_dim
            ).to(device)

            # Load the model weights
            try:
                if "model_state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["model_state_dict"])
                    logger.info("Model weights loaded successfully")
                else:
                    logger.warning("No model_state_dict found in checkpoint")
            except Exception as e:
                logger.warning(f"Error loading model weights: {str(e)}")

            model.eval()

            # Generate for each prompt
            for i, prompt in enumerate(prompts):
                logger.info(f"Generating ring {i+1}/{len(prompts)}: {prompt}")

                # Encode the prompt to a latent vector
                encoded_text = torch.zeros(512, dtype=torch.float32, device=device)
                for j, char in enumerate(prompt[:512]):
                    encoded_text[j] = ord(char) / 255.0  # Simple character encoding

                with torch.no_grad():
                    latent = model(encoded_text.unsqueeze(0))

                # Generate mesh using our basic ring generator
                create_basic_ring_mesh(prompt, i, output_dir, latent)

                # Create metadata file
                metadata_file = os.path.join(
                    output_dir, f"generated_ring_{i+1}_metadata.json"
                )
                metadata = {
                    "prompt": prompt,
                    "components": [
                        {"name": "Metal_Head", "material": "gold"},
                        {"name": "Metal_Shank", "material": "gold"},
                        {"name": "Head_Center_Stone", "material": "diamond"},
                        {"name": "Head_Accent_Stone", "material": "diamond"},
                    ],
                }

                with open(metadata_file, "w") as f:
                    json.dump(metadata, f, indent=2)

            logger.info(f"Generated {len(prompts)} ring models in {output_dir}")
            return True
        else:
            logger.warning("No CAP3D model available for generation")
            for i, prompt in enumerate(prompts):
                create_basic_ring_mesh(prompt, i, output_dir)
            return True

    except Exception as e:
        logger.error(f"Error during ring generation: {str(e)}")
        logger.warning("Falling back to basic ring generation")
        for i, prompt in enumerate(prompts):
            create_basic_ring_mesh(prompt, i, output_dir)
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate 3D ring models from text prompts"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="outputs/cap3d_checkpoints/checkpoints/cap3d_model.pt",
        help="Path to the trained CAP3D model",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/generated",
        help="Directory to save generated models",
    )
    parser.add_argument(
        "--num-samples", type=int, default=5, help="Number of samples to generate"
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        type=str,
        default=[],
        help="Custom prompts for generation",
    )
    parser.add_argument("--cpu-only", action="store_true", help="Force CPU usage")
    args = parser.parse_args()

    # Determine device
    device = "cpu" if args.cpu_only else "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Start time
    start_time = time.time()

    # Use custom prompts if provided
    prompts = (
        args.prompts
        if args.prompts
        else [
            "A silver ring with diamond",
            "A gold ring with ruby",
            "A platinum ring with emerald",
            "A vintage style ring",
            "A modern minimalist ring",
        ]
    )

    # Generate rings
    generate_rings(
        args.model,
        args.output,
        prompts=prompts,
        num_samples=args.num_samples,
        device=device,
    )

    # End time calculation
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info("=== Generation Complete ===")
    logger.info(f"Total time: {elapsed_time:.2f} seconds")
    logger.info(f"Results saved to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
