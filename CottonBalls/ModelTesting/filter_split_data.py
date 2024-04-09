import os
import glob
from collections import Counter
import shutil
from sklearn.model_selection import train_test_split
import pandas as pd
import cv2


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
    data_dict = {}
    subfolders = [f.path for f in os.scandir(base_folder) if f.is_dir()]

    # Iterate through the subfolders and count files in those starting with "images_"
    for folder in subfolders:
        folder_name = os.path.basename(folder)
        if folder_name.startswith("images_"):
            file_count = len(glob.glob(os.path.join(folder, '*')))
            image_files = glob.glob(os.path.join(folder, '*.png'))
            ID = os.path.basename(image_files[0])[4:10]
            data_dict[folder_name] = {
                "FileCount": file_count,
                "ID": ID
            }

            print(f"Folder: {folder_name}. # of Files: {file_count}. ID: {ID}")
    
    sorted_data_dict = dict(sorted(data_dict.items(), key=lambda item: int(item[1]["ID"])))

    return sorted_data_dict

def export_dict_to_txt(data_dict, file_path):
    try:
        with open(file_path, 'w') as file:
            for folder_name, data in data_dict.items():
                line = f"{folder_name},{data['FileCount']},{data['ID']}\n"
                file.write(line)
        print(f"Dictionary exported to '{file_path}")
    except Exception as e:
        print(f"Error exporting dictionary: {str(e)}")

def read_dict_from_txt(file_path):
    try:
        data_dict = {}
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.strip().split(',')
                if len(parts) == 3:
                    folder_name, file_count, ID = parts
                    data_dict[folder_name] = {
                        "FileCount": int(file_count),
                        "ID": ID
                    }
        print(f"Dictionary loaded from '{file_path}'")
        return data_dict
    except Exception as e:
        print(f"Error loading dictionary: {str(e)}")
        return None

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


def rename_files(images_and_annotations_dir, images_and_annotations_renamed_dir):

    shutil.copytree(images_and_annotations_dir, images_and_annotations_renamed_dir)

    dir = images_and_annotations_renamed_dir

    dates = [20230405, 20230412, 20230419, 20230426, 20230503]

    cotton_ball_size_dict = {
        "0": "1MM",
        "1": "2MM",
        "2": "3MM",
        "3": "5MM",
        "4": "10MM",
        "5": "15MM",
        "6": "20MM",
        "999": "0MM"
    }

    for date in dates:
        images_dir = dir + str(date) + "_images/"
        annotations_dir = dir + str(date) + "_annotations/"

        images = glob.glob(os.path.join(images_dir, '*.png'))
        annotations = glob.glob(os.path.join(annotations_dir, '*.txt'))
        mylist = []
        for annotation in annotations:
            with open(annotation, 'r') as file:
                content = file.read()
            lines = content.strip().split('\n')
            print(lines)
            if lines == ['']:
                classes = [999]
            else:
                classes = [line.split()[0] for line in lines]
            counts = Counter(classes)
            sorted_counts = dict(sorted(counts.items()))

            counts_string = [f"{count}_{cotton_ball_size_dict[f'{element}']}" for element, count in sorted_counts.items()]
            new_filename_string = "_".join(counts_string)
            if new_filename_string == "1_0MM":
                new_filename_string = "0_0MM"
            
            old_filename = os.path.basename(annotation)
            new_filename = os.path.join(new_filename_string + "_" + old_filename)
            new_filename_path = os.path.join(annotations_dir, new_filename)
            os.rename(annotation, new_filename_path)

            image_path = os.path.join(images_dir, old_filename.replace(".txt", ".png"))
            new_image_path = os.path.join(images_dir, new_filename.replace(".txt", ".png"))
            os.rename(image_path, new_image_path)

def partition(images_and_annotations_renamed_dir, filter_one_data_excel_path=0, filter_two_data_excel_path=0):
    
    dir = "/Users/jayvik/Desktop/Data/"

    os.makedirs(dir + "partitioned_data/all_images")
    os.makedirs(dir + "partitioned_data/all_annotations")

    #copy images and annotations into folders called "all_images" and "all_annotations"
    dates = [20230405, 20230412, 20230419, 20230426, 20230503]
    for date in dates:
        images_dir = images_and_annotations_renamed_dir + str(date) + "_images/"
        annotations_dir = images_and_annotations_renamed_dir + str(date) + "_annotations/"

        images = glob.glob(os.path.join(images_dir, '*.png'))
        annotations = glob.glob(os.path.join(annotations_dir, '*.txt'))

        for image in images:
            shutil.copy(image, dir + "partitioned_data/all_images/")
        for annotation in annotations:
            shutil.copy(annotation, dir + "partitioned_data/all_annotations/")

    #first round of filtering - gets rid of images that are bad or okay
    if filter_one_data_excel_path != 0:
        if not os.path.exists(dir + "partitioned_data/filter_one_images/"):
            shutil.copytree(dir + "partitioned_data/all_images/", dir + "partitioned_data/filter_one_images/")
        if not os.path.exists(dir + "partitioned_data/filter_one_annotations/"):
            shutil.copytree(dir + "partitioned_data/all_annotations/", dir + "partitioned_data/filter_one_annotations/")


        images = glob.glob(os.path.join(dir + "partitioned_data/filter_one_images/", '*.png'))

        
        df = pd.read_excel(filter_one_data_excel_path)
        bad_images = df['Bad'].tolist()
        okay_images = df['Okay'].tolist()
        round3_images = df['Round3'].tolist()
        remove_list = bad_images + okay_images + round3_images
        for image in images:
            filepath = os.path.basename(image)
            if filepath[-19:-4] in remove_list:
                os.remove(image)
                os.remove(dir + "partitioned_data/filter_one_annotations/" + filepath.replace(".png", ".txt"))

    #second round of filtering - gets rid of images with incorrect number of objects
    if filter_two_data_excel_path != 0:
        if not os.path.exists(dir + "partitioned_data/filter_two_images/"):
            shutil.copytree(dir + "partitioned_data/filter_one_images/", dir + "partitioned_data/filter_two_images/")
        if not os.path.exists(dir + "partitioned_data/filter_two_annotations/"):
            shutil.copytree(dir + "partitioned_data/filter_one_annotations/", dir + "partitioned_data/filter_two_annotations/")
        
        images = glob.glob(os.path.join(dir + "partitioned_data/filter_two_images/", '*.png'))

        compare_dict = {}
        df = pd.read_excel(filter_two_data_excel_path)
        ids = df['ID'].tolist()
        target_objects = df['Number of objects'].tolist()
        for i in range(len(ids)):
            compare_dict[int(ids[i])] = str(target_objects[i])
            # print(ids[i], target_objects[i])

        
        for image in images:
            filepath = os.path.basename(image)
            image_id = int(filepath[-15:-9])
            labelled_objects = str(filepath[:-20])
            # print(labelled_objects)

            if labelled_objects != compare_dict[image_id]:
                os.remove(image)
                os.remove(dir + "partitioned_data/filter_two_annotations/" + filepath.replace(".png", ".txt"))

    filter_two_images = glob.glob(os.path.join(dir + "partitioned_data/filter_two_images/", '*.png'))
    filter_two_annotations = glob.glob(os.path.join(dir + "partitioned_data/filter_two_annotations/", '*.txt'))

    #create folders for train, test, and val
    image_train_folder = dir + "partitioned_data/images/train"
    image_test_folder = dir + "partitioned_data/images/test"
    image_val_folder = dir + "partitioned_data/images/val"
    annotation_train_folder = dir + "partitioned_data/annotations/train"
    annotation_test_folder = dir + "partitioned_data/annotations/test"
    annotation_val_folder = dir + "partitioned_data/annotations/val"

    os.makedirs(image_train_folder)
    os.makedirs(image_test_folder)
    os.makedirs(image_val_folder)
    os.makedirs(annotation_train_folder)
    os.makedirs(annotation_test_folder)
    os.makedirs(annotation_val_folder)

    images.sort()
    annotations.sort()

    #CHECK IF IMAGES AND ANNOTATIONS MATCH
    def check_images_annotations(images_folder, annotations_folder):
        for i in images_folder:
            i = os.path.basename(i)
            a = i.replace(".png", ".txt")
            a_list = []
            for j in annotations_folder:
                a_list.append(os.path.basename(j))
            
            if a not in a_list:
                print("ERROR", a)
            else:
                print("SUCCESS")

    # check_images_annotations(filter_two_images, filter_two_annotations)

    #split data into train, test, and val
    train_images, test_images = train_test_split(filter_two_images, test_size=0.15, random_state=42)
    train_images, val_images = train_test_split(train_images, test_size=0.2, random_state=42)

    #print(len(train_images), len(test_images), len(val_images))

    for train_image in train_images:
        shutil.copy(train_image, image_train_folder)
        train_annotations = train_image.replace(".png", ".txt").replace("images", "annotations")
        shutil.copy(train_annotations, annotation_train_folder)
    for test_image in test_images:
        shutil.copy(test_image, image_test_folder)
        test_annotations = test_image.replace(".png", ".txt").replace("images", "annotations")
        shutil.copy(test_annotations, annotation_test_folder)
    for val_image in val_images:
        shutil.copy(val_image, image_val_folder)
        val_annotations = val_image.replace(".png", ".txt").replace("images", "annotations")
        shutil.copy(val_annotations, annotation_val_folder)
    
    #check train, test, and val images and annotations match
    # check_images_annotations(glob.glob(os.path.join(image_train_folder, '*.png')), glob.glob(os.path.join(annotation_train_folder, '*.txt')))
    # check_images_annotations(glob.glob(os.path.join(image_test_folder, '*.png')), glob.glob(os.path.join(annotation_test_folder, '*.txt')))
    # check_images_annotations(glob.glob(os.path.join(image_val_folder, '*.png')), glob.glob(os.path.join(annotation_val_folder, '*.txt')))

if __name__ == "__main__":
    images_and_annotations_dir = "/Users/jayvik/Desktop/Data/images_and_annotations/"

    filter_one_data_excel_path = "/Users/jayvik/Desktop/Data/FilterOne_CottonBallData_HEPIUS.xlsx"
    filter_two_data_excel_path = "/Users/jayvik/Desktop/Data/FilterTwo_CottonBallData_HEPIUS.xlsx"
    images_and_annotations_renamed_dir = "/Users/jayvik/Desktop/Data/images_and_annotations_renamed/"

    #rename_files(images_and_annotations_dir, images_and_annotations_renamed_dir)
    partition(images_and_annotations_renamed_dir, filter_one_data_excel_path, filter_two_data_excel_path)