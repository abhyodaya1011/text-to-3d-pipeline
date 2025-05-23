#!/usr/bin/env python
"""
Simple setup script for accessing files from a shared Google Drive folder in Colab.
This script provides a direct solution that works reliably in Google Colab.
"""

def setup_from_shared_folder():
    """
    Set up the RingGen pipeline using files from a shared Google Drive folder in Colab.
    This function mounts Google Drive and provides instructions for manual copying.
    """
    # Import Colab-specific modules
    try:
        from google.colab import drive
        print("Mounting Google Drive...")
        drive.mount('/content/drive')
        
        # Create necessary directories
        import os
        os.makedirs("data/rings", exist_ok=True)
        os.makedirs("shap_e_model_cache", exist_ok=True)
        os.makedirs("outputs/training", exist_ok=True)
        os.makedirs("outputs/generated", exist_ok=True)
        
        print("\n=== SETUP INSTRUCTIONS ===")
        print("1. Navigate to the shared folder in Google Drive")
        print("2. Copy all OBJ files to the 'data/rings/' directory")
        print("3. Copy all PT files (model cache) to the 'shap_e_model_cache/' directory")
        print("4. Copy any other files to the appropriate directories")
        print("\nYou can also use the following code to copy files automatically:")
        print("```python")
        print("import shutil")
        print("import glob")
        print("")
        print("# Replace with the actual path to your shared folder")
        print("shared_folder = '/content/drive/MyDrive/path/to/shared/folder'")
        print("")
        print("# Copy OBJ files to data/rings/")
        print("for obj_file in glob.glob(shared_folder + '/**/*.obj', recursive=True):")
        print("    shutil.copy(obj_file, 'data/rings/' + os.path.basename(obj_file))")
        print("    print(f'Copied {os.path.basename(obj_file)} to data/rings/')")
        print("")
        print("# Copy PT files to shap_e_model_cache/")
        print("for pt_file in glob.glob(shared_folder + '/**/*.pt', recursive=True):")
        print("    shutil.copy(pt_file, 'shap_e_model_cache/' + os.path.basename(pt_file))")
        print("    print(f'Copied {os.path.basename(pt_file)} to shap_e_model_cache/')")
        print("```")
        
        return True
    except ImportError:
        print("This script is designed to be run in Google Colab.")
        print("Please run this script in a Google Colab notebook.")
        return False
    except Exception as e:
        print(f"Error setting up: {str(e)}")
        return False

if __name__ == "__main__":
    setup_from_shared_folder()
