import os
import zipfile

# Set the directory path where the folders are located
base_dir = os.path.expanduser('~/Developer/VisionLangAnnotate/output/dashcam_videos/onevideo_yolo11l_v2')

# Iterate over all entries in the directory
for folder_name in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder_name)
    if os.path.isdir(folder_path):
        zip_path = os.path.join(base_dir, folder_name + '.zip')
        print(f'Zipping {folder_name} -> {zip_path}')
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(folder_path):
                for file in files:
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, start=folder_path)
                    zipf.write(full_path, arcname=os.path.join(folder_name, rel_path))