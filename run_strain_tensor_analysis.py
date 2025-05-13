import os
import re
import subprocess
import shutil

# +
# run strain tensor analysis

slice_depth_list = [-8.0]
# Ensure the main directory exists
if os.path.isdir('hany_models'):
    # Iterate over subdirectories
    for slice_depth in slice_depth_list:
        for folder in os.listdir('hany_models'):
            folder_path = os.path.join('hany_models', folder)
            if os.path.isdir(folder_path):
                print("Processing folder:", folder_path)
                
                # Iterate over files in the subdirectory
                for file in os.listdir(folder_path):
                    # Match files with names like 'velocityField-12.h5', 'velocityField-19.h5', etc.
                    match = re.match(r"velocityField-(\d+)\.h5", file)
                    if match:
                        velocity_number = match.group(1)
                        velocity_file_path = os.path.join(folder_path, file)
                        print("Found velocity file:", velocity_file_path, "with velocity number:", velocity_number, "at depth:", slice_depth)
                        
                        # Define the command as a list of arguments
                        command = [
                                    "python", "strain_tensor_analysis.py",
                                    "--input_dir", folder_path+'/',
                                    "--slice_depth", str(slice_depth),
                                    "--vel_file_no", str(velocity_number)
                                ]
                        # Execute the command
                        subprocess.run(command)


# -

def copy_folder_and_remove_h5(src_folder, dest_folder):
    # Copy the entire source folder to the destination folder.
    # If the destination folder exists, remove it first (optional).
    if os.path.exists(dest_folder):
        shutil.rmtree(dest_folder)
    
    shutil.copytree(src_folder, dest_folder)
    print(f"Copied {src_folder} to {dest_folder}")

    # Walk through the copied folder and remove all .h5 files.
    for root, dirs, files in os.walk(dest_folder):
        for file in files:
            if file.endswith('.h5'):
                file_path = os.path.join(root, file)
                os.remove(file_path)
                print(f"Deleted: {file_path}")


def copy_files_with_minus8(src_folder, dest_folder):
    """
    Copy only files with '_-8' in their basename from src_folder into dest_folder,
    preserving the directory tree.
    """
    # If the destination folder exists, remove it first
    if os.path.exists(dest_folder):
        shutil.rmtree(dest_folder)

    # Walk source tree
    for root, dirs, files in os.walk(src_folder):
        # Compute path relative to src_folder
        rel_dir = os.path.relpath(root, src_folder)
        # Prepare corresponding destination directory
        dest_dir = os.path.join(dest_folder, rel_dir)
        os.makedirs(dest_dir, exist_ok=True)

        # Copy only files matching *_-8*
        for fname in files:
            if "_-8" in fname:
                src_path = os.path.join(root, fname)
                dest_path = os.path.join(dest_dir, fname)
                shutil.copy2(src_path, dest_path)
                print(f"Copied: {src_path} -> {dest_path}")



# Example usage:
src = "./hany_models"   # Replace with your source folder path
dest = "./hany_models_cp"  # Replace with your destination folder path
# copy_folder_and_remove_h5(src, dest)
copy_files_with_minus8(src, dest)


