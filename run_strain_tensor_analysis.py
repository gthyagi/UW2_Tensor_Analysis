import os
import re
import subprocess
import shutil

# +
# run strain tensor analysis

slice_depth_list = [0.0, -4.0, -8.0]
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


# Example usage:
src = "./hany_models"   # Replace with your source folder path
dest = "./hany_models_cp"  # Replace with your destination folder path
copy_folder_and_remove_h5(src, dest)


