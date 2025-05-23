#!/usr/bin/env python
"""
Utility functions for accessing files from shared Google Drive links and folders.
This module provides functions to download files from shared Google Drive links
and set up the necessary directory structure for the RingGen pipeline.
"""

import os
import re
import json
import requests
import shutil
import subprocess
import time
from pathlib import Path
from tqdm import tqdm

def extract_file_id_from_drive_link(link):
    """Extract the file ID from a Google Drive link."""
    # Pattern for Google Drive links
    patterns = [
        r'drive\.google\.com/file/d/([a-zA-Z0-9_-]+)',  # For file links
        r'drive\.google\.com/open\?id=([a-zA-Z0-9_-]+)',  # For open links
        r'drive\.google\.com/drive/folders/([a-zA-Z0-9_-]+)',  # For folder links
        r'id=([a-zA-Z0-9_-]+)'  # Generic pattern
    ]
    
    for pattern in patterns:
        match = re.search(pattern, link)
        if match:
            return match.group(1)
    
    return None

def download_file_from_drive(file_id, destination):
    """Download a file from Google Drive using file ID."""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    # URL for downloading from Google Drive
    URL = "https://drive.google.com/uc?export=download"
    
    # First request to get the confirmation token
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = None
    
    # Check if there's a download warning (for large files)
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break
    
    # If we have a token, we need to confirm the download
    if token:
        params = {'id': file_id, 'confirm': token}
    else:
        params = {'id': file_id}
    
    # Download the file with progress bar
    response = session.get(URL, params=params, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as f, tqdm(
            desc=os.path.basename(destination),
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                size = f.write(chunk)
                bar.update(size)
    
    return os.path.exists(destination)

def download_from_drive_link(drive_link, destination):
    """Download a file from a Google Drive link to a destination path."""
    file_id = extract_file_id_from_drive_link(drive_link)
    if not file_id:
        print(f"Could not extract file ID from link: {drive_link}")
        return False
    
    return download_file_from_drive(file_id, destination)

def list_files_in_folder(folder_id):
    """List all files in a Google Drive folder."""
    try:
        # Try to use gdown if available
        import gdown
        url = f"https://drive.google.com/drive/folders/{folder_id}"
        files = gdown.download_folder(url, quiet=False, use_cookies=False)
        return files
    except ImportError:
        print("gdown not found. Installing...")
        subprocess.run(["pip", "install", "gdown"], check=True)
        import gdown
        url = f"https://drive.google.com/drive/folders/{folder_id}"
        files = gdown.download_folder(url, quiet=False, use_cookies=False)
        return files

def setup_from_shared_folder(folder_link):
    """
    Set up the RingGen pipeline using files from a shared Google Drive folder.
    
    Args:
        folder_link: Link to a shared Google Drive folder
    
    Returns:
        bool: True if setup was successful, False otherwise
    """
    folder_id = extract_file_id_from_drive_link(folder_link)
    if not folder_id:
        print(f"Could not extract folder ID from link: {folder_link}")
        return False
    
    print(f"Setting up from shared folder: {folder_link}")
    print(f"Folder ID: {folder_id}")
    
    # Create necessary directories
    os.makedirs("data/rings", exist_ok=True)
    os.makedirs("shap_e_model_cache", exist_ok=True)
    os.makedirs("outputs/training", exist_ok=True)
    os.makedirs("outputs/generated", exist_ok=True)
    
    # Download all files from the folder
    try:
        print("Downloading files from shared folder...")
        downloaded_files = list_files_in_folder(folder_id)
        
        if not downloaded_files:
            print("No files were downloaded from the shared folder.")
            return False
        
        print(f"Downloaded {len(downloaded_files)} files from the shared folder.")
        
        # Process downloaded files
        for file in downloaded_files:
            filename = os.path.basename(file)
            
            # Handle OBJ files (ring models)
            if filename.endswith('.obj'):
                dest_path = os.path.join("data/rings", filename)
                shutil.move(file, dest_path)
                print(f"Moved {filename} to data/rings/")
            
            # Handle PT files (model cache)
            elif filename.endswith('.pt'):
                dest_path = os.path.join("shap_e_model_cache", filename)
                shutil.move(file, dest_path)
                print(f"Moved {filename} to shap_e_model_cache/")
            
            # Handle ZIP files
            elif filename.endswith('.zip'):
                if 'ring' in filename.lower() or 'data' in filename.lower():
                    # Ring data zip
                    print(f"Extracting {filename} to data/rings/")
                    shutil.unpack_archive(file, "data/rings/")
                elif 'model' in filename.lower() or 'cache' in filename.lower() or 'shap' in filename.lower():
                    # Model cache zip
                    print(f"Extracting {filename} to shap_e_model_cache/")
                    shutil.unpack_archive(file, "shap_e_model_cache/")
                elif 'output' in filename.lower() or 'result' in filename.lower():
                    # Outputs zip
                    print(f"Extracting {filename} to outputs/")
                    shutil.unpack_archive(file, "outputs/")
                else:
                    # Unknown zip, extract to current directory
                    print(f"Extracting {filename} to current directory")
                    shutil.unpack_archive(file, "./")
                
                # Remove the zip file after extraction
                os.remove(file)
            
            # Other files
            else:
                print(f"Keeping {filename} in current directory")
        
        return True
    
    except Exception as e:
        print(f"Error setting up from shared folder: {str(e)}")
        return False

def setup_from_shared_drive(ring_data_link=None, model_cache_link=None, output_link=None):
    """
    Set up the RingGen pipeline using files from shared Google Drive links.
    
    Args:
        ring_data_link: Link to a zip file containing ring data
        model_cache_link: Link to a zip file containing model cache
        output_link: Link to a zip file containing outputs
    
    Returns:
        bool: True if setup was successful, False otherwise
    """
    # Create necessary directories
    os.makedirs("data/rings", exist_ok=True)
    os.makedirs("shap_e_model_cache", exist_ok=True)
    os.makedirs("outputs/training", exist_ok=True)
    os.makedirs("outputs/generated", exist_ok=True)
    
    success = True
    
    # Download and extract ring data if provided
    if ring_data_link:
        print(f"Downloading ring data from shared link...")
        zip_path = "data/ring_data.zip"
        if download_from_drive_link(ring_data_link, zip_path):
            print(f"Extracting ring data to data/rings/...")
            shutil.unpack_archive(zip_path, "data/rings/")
            os.remove(zip_path)
        else:
            print(f"Failed to download ring data from {ring_data_link}")
            success = False
    
    # Download and extract model cache if provided
    if model_cache_link:
        print(f"Downloading model cache from shared link...")
        zip_path = "model_cache.zip"
        if download_from_drive_link(model_cache_link, zip_path):
            print(f"Extracting model cache to shap_e_model_cache/...")
            shutil.unpack_archive(zip_path, "shap_e_model_cache/")
            os.remove(zip_path)
        else:
            print(f"Failed to download model cache from {model_cache_link}")
            success = False
    
    # Download and extract outputs if provided
    if output_link:
        print(f"Downloading outputs from shared link...")
        zip_path = "outputs.zip"
        if download_from_drive_link(output_link, zip_path):
            print(f"Extracting outputs to outputs/...")
            shutil.unpack_archive(zip_path, "outputs/")
            os.remove(zip_path)
        else:
            print(f"Failed to download outputs from {output_link}")
            success = False
    
    return success

# Example usage in Colab:
"""
from drive_utils import setup_from_shared_drive

# Replace these with your actual shared Google Drive links
ring_data_link = "https://drive.google.com/file/d/YOUR_FILE_ID/view?usp=sharing"
model_cache_link = "https://drive.google.com/file/d/YOUR_FILE_ID/view?usp=sharing"
output_link = "https://drive.google.com/file/d/YOUR_FILE_ID/view?usp=sharing"

setup_from_shared_drive(ring_data_link, model_cache_link, output_link)
"""

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Set up RingGen from shared Google Drive links")
    parser.add_argument("--ring-data", type=str, help="Link to ring data zip file")
    parser.add_argument("--model-cache", type=str, help="Link to model cache zip file")
    parser.add_argument("--outputs", type=str, help="Link to outputs zip file")
    
    args = parser.parse_args()
    
    setup_from_shared_drive(args.ring_data, args.model_cache, args.outputs)
