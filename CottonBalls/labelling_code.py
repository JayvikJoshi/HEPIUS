import os
import glob
from collections import Counter
import shutil


def get_txt_file_paths(folder_path):
    return glob.glob(os.path.join(folder_path, '*.txt'))

def rewrite_txt(file_path, replacement_text):
    with open(file_path, 'w') as file:
        file.write(replacement_text)

def append_txt(file_path, append_text):
    with open(file_path, 'a') as file:
        file.write(append_text)

def prepend_txt(file_path, prepend_text):
    with open(file_path, 'r+') as file:
        content = file.read()
        file.seek(0, 0)
        file.write(prepend_text.rstrip('\r\n') + '\n' + content)

def isolate_txt(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    with open(file_path, 'w') as file:
        data_to_write = []
        for line in lines:
            parts = line.split()
            isolated_data = ' '.join(parts[1:])
            brackets_removed = isolated_data.replace('[', '').replace(']', '')
            data_to_write.append(brackets_removed + '\n')
        file.writelines(data_to_write)

def rename_file_with_content(file_path):
    print(f"file_path: {file_path}")

    with open(file_path, 'r') as file:
        content = file.read()
    lines = content.strip().split('\n')
    ball_info = [line.split()[0] for line in lines]

    element_counts = Counter(ball_info)
    sorted_element_counts = dict(sorted(element_counts.items()))

    element_strings = [f"{count}_{element}mm" for element, count in sorted_element_counts.items()]

    new_filename = "_".join(element_strings) + ".txt"


    directory = os.path.dirname(file_path)
    new_file_path = os.path.join(directory, new_filename)
    os.rename(file_path, new_file_path)
    return new_file_path

def count_files_in_image_folders(base_folder):
    # Get a list of all subdirectories in the base folder
    subfolders = [f.path for f in os.scandir(base_folder) if f.is_dir()]

    # Iterate through the subfolders and count files in those starting with "images_"
    for folder in subfolders:
        folder_name = os.path.basename(folder)
        if folder_name.startswith("images_"):
            file_count = len(glob.glob(os.path.join(folder, '*')))
            image_files = glob.glob(os.path.join(folder, '*.png'))
            first_name = os.path.basename(image_files[0])[4:10]
            print(f"Folder: {folder_name}. # of Files: {file_count}. ID: {first_name}")

def copy_images_to_combined_folder(base_folder):
    # Create the "combined" folder if it doesn't exist
    combined_folder = os.path.join(base_folder, "combined")
    if not os.path.exists(combined_folder):
        os.makedirs(combined_folder)

    # Get a list of all subdirectories in the base folder
    subfolders = [f.path for f in os.scandir(base_folder) if f.is_dir()]

    # Iterate through the subfolders and copy images from those starting with "images_"
    for folder in subfolders:
        folder_name = os.path.basename(folder)
        if folder_name.startswith("images_"):
            image_files = glob.glob(os.path.join(folder, '*.png'))  # Change the extension as needed
            for image_file in image_files:
                # Create a new file path in the "combined" folder
                new_file_path = os.path.join(combined_folder, os.path.basename(image_file))
                # Copy the image file to the "combined" folder
                shutil.copy(image_file, new_file_path)
                print(f"Copied '{image_file}' to 'combined' folder.")

if __name__ == "__main__":
    folder_path="/Users/jayvik/Desktop/final/20230405"
    count_files_in_image_folders(folder_path)