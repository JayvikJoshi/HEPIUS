import os
import glob
from collections import Counter


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


if __name__ == "__main__":
    file_path="/Users/jayvik/Desktop/test_data.txt"
    
    new_file_path=rename_file_with_content(file_path)

    isolate_txt(file_path)