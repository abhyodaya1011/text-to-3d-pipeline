#!/usr/bin/env python
"""
Setup script for accessing files from a shared Google Drive folder in Colab.
This script is designed to be used in Google Colab to access files from a
shared Google Drive folder for the RingGen pipeline.
"""

import os
import re
import sys
import shutil
import subprocess
from pathlib import Path
from google.colab import drive

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

def setup_from_shared_folder(folder_link):
    """
    Set up the RingGen pipeline using files from a shared Google Drive folder in Colab.
    
    Args:
        folder_link: Link to a shared Google Drive folder
    
    Returns:
        bool: True if setup was successful, False otherwise
    """
    try:
        # Mount Google Drive
        print("Mounting Google Drive...")
        drive.mount('/content/drive')
        
        # Extract folder ID from link
        folder_id = extract_file_id_from_drive_link(folder_link)
        if not folder_id:
            print(f"Could not extract folder ID from link: {folder_link}")
            return False
        
        print(f"Setting up from shared folder with ID: {folder_id}")
        
        # Create necessary directories
        os.makedirs("data/rings", exist_ok=True)
        os.makedirs("shap_e_model_cache", exist_ok=True)
        os.makedirs("outputs/training", exist_ok=True)
        os.makedirs("outputs/generated", exist_ok=True)
        
        # Generate a script to find and process files
        script = f"""
# This script will copy files from the shared folder to the appropriate directories
import os
import glob
import shutil
from google.colab import drive

# Ensure drive is mounted
if not os.path.exists('/content/drive/MyDrive'):
    drive.mount('/content/drive')

# Function to search for the shared folder in Google Drive
def find_shared_folder(folder_id):
    # Check in Shared drives
    shared_drives_path = '/content/drive/Shareddrives'
    if os.path.exists(shared_drives_path):
        for root, dirs, _ in os.walk(shared_drives_path):
            for dir in dirs:
                if folder_id in dir:
                    return os.path.join(root, dir)
    
    # Check in My Drive shared folders
    my_drive_path = '/content/drive/MyDrive'
    for root, dirs, _ in os.walk(my_drive_path):
        for dir in dirs:
            if folder_id in dir:
                return os.path.join(root, dir)
    
    # If not found, try direct path for shared folders
    direct_path = f'/content/drive/MyDrive/Shared with me/{folder_id}'
    if os.path.exists(direct_path):
        return direct_path
        
    # Try another common path
    alt_path = f'/content/drive/Shareddrives/{folder_id}'
    if os.path.exists(alt_path):
        return alt_path
    
    return None

# Find the shared folder
folder_path = find_shared_folder('{folder_id}')

if not folder_path:
    print(f"Could not find shared folder with ID: {folder_id} in Google Drive.")
    print("Please make sure the folder is shared with you and accessible in Google Drive.")
    print("You might need to manually navigate to the folder in the Google Drive UI first.")
    
    # Try a more direct approach - use the folder ID directly
    folder_path = f'/content/drive/MyDrive/{folder_id}'
    if not os.path.exists(folder_path):
        folder_path = f'/content/drive/Shareddrives/{folder_id}'
    if not os.path.exists(folder_path):
        # Last resort - ask user to provide path
        print("\\nPlease manually navigate to the shared folder in Google Drive,")
        print("then enter the full path to the folder:")
        folder_path = input()

if not os.path.exists(folder_path):
    print(f"Could not access folder at {folder_path}")
    exit(1)
    
print(f"Found shared folder at: {folder_path}")

# Process all files in the folder
file_count = 0

# Process OBJ files (ring models)
obj_files = glob.glob(os.path.join(folder_path, '**', '*.obj'), recursive=True)
for obj_file in obj_files:
    dest_path = os.path.join('data/rings', os.path.basename(obj_file))
    shutil.copy(obj_file, dest_path)
    print(f"Copied {os.path.basename(obj_file)} to data/rings/")
    file_count += 1

# Process PT files (model cache)
pt_files = glob.glob(os.path.join(folder_path, '**', '*.pt'), recursive=True)
for pt_file in pt_files:
    dest_path = os.path.join('shap_e_model_cache', os.path.basename(pt_file))
    shutil.copy(pt_file, dest_path)
    print(f"Copied {os.path.basename(pt_file)} to shap_e_model_cache/")
    file_count += 1

# Process ZIP files
zip_files = glob.glob(os.path.join(folder_path, '**', '*.zip'), recursive=True)
for zip_file in zip_files:
    filename = os.path.basename(zip_file)
    if 'ring' in filename.lower() or 'data' in filename.lower():
        # Ring data zip
        print(f"Extracting {filename} to data/rings/")
        shutil.unpack_archive(zip_file, "data/rings/")
    elif 'model' in filename.lower() or 'cache' in filename.lower() or 'shap' in filename.lower():
        # Model cache zip
        print(f"Extracting {filename} to shap_e_model_cache/")
        shutil.unpack_archive(zip_file, "shap_e_model_cache/")
    elif 'output' in filename.lower() or 'result' in filename.lower():
        # Outputs zip
        print(f"Extracting {filename} to outputs/")
        shutil.unpack_archive(zip_file, "outputs/")
    else:
        # Unknown zip, extract to current directory
        print(f"Extracting {filename} to current directory")
        shutil.unpack_archive(zip_file, "./")
    file_count += 1

# Process other files
all_files = glob.glob(os.path.join(folder_path, '*'))
for file_path in all_files:
    if os.path.isfile(file_path) and not file_path.endswith(('.obj', '.pt', '.zip')):
        shutil.copy(file_path, './')
        print(f"Copied {os.path.basename(file_path)} to current directory")
        file_count += 1

print(f"\\nProcessed {file_count} files from the shared folder.")
"""
        
        # Write the script to a file
        with open('process_shared_folder.py', 'w') as f:
            f.write(script)
        
        # Execute the script
        print("\nExecuting script to process shared folder...")
        subprocess.run([sys.executable, 'process_shared_folder.py'], check=True)
        
        return True
    
    except Exception as e:
        print(f"Error setting up from shared folder in Colab: {str(e)}")
        return False

if __name__ == "__main__":
    # If run directly, get the folder link from command line arguments
    if len(sys.argv) > 1:
        folder_link = sys.argv[1]
        setup_from_shared_folder(folder_link)
    else:
        print("Please provide a shared folder link as an argument.")
        print("Example: python colab_drive_setup.py https://drive.google.com/drive/folders/YOUR_FOLDER_ID")
