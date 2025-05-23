# RingGen: Streamlined Text-to-3D Ring Generation Pipeline

This repository contains a streamlined version of the RingGen pipeline for training and generating 3D ring models using Shap-E and CAP3D. It's specifically designed to work efficiently on Google Colab with GPU acceleration.

## Project Structure

```
text-to-3d-pipeline/
├── ringgen/                  # Core package with minimal modules
├── ringgen_colab.ipynb       # The Colab notebook for end-to-end pipeline
├── train_with_labeled_data.py # Main training script
├── generate_rings.py         # Generation script
├── setup_shap_e.py           # Setup script for Shap-E
├── setup.py                  # Simplified package installation
└── .gitignore                # Excludes large files and generated directories
```

## How to Use

1. **Upload to Google Colab**: Upload the `ringgen_colab.ipynb` notebook to Google Colab
2. **Run the Notebook**: Follow the step-by-step instructions in the notebook
3. **Store Data on Google Drive**: Use Google Drive for storing training data and results

## Features

- **GPU-Accelerated Training**: Automatically uses GPU when available
- **Interactive Prompts**: Customize training parameters and generation prompts
- **Visualization Tools**: View generated rings with both Matplotlib and Plotly
- **Google Drive Integration**: Seamlessly access and save data to Google Drive

## Requirements

The notebook will automatically install all required dependencies, including:
- PyTorch
- Trimesh
- Numpy
- Matplotlib
- Plotly
- Shap-E (installed automatically)

## Workflow

1. **Environment Setup**: Check GPU availability and install dependencies
2. **Data Preparation**: Access ring models from Google Drive
3. **Training** (Optional): Train Shap-E and CAP3D models on your data
4. **Generation**: Create 3D ring models from text prompts
5. **Visualization**: View and analyze the generated models
6. **Export**: Save results to Google Drive and download as needed

## Notes

- This is a streamlined version of the original project, focusing only on the essential components needed for practical training and generation
- Large files like training data and model caches should be stored on Google Drive, not in this repository
