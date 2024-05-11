import os
import pandas as pd

original_dicoms = 1
original_pngs = '/Users/jayvik/Desktop/CottonBall_Data/original_pngs'
excel_path = '/Users/jayvik/Desktop/CottonBall_Data/cottonball_info.xlsx'

df = pd.read_excel(excel_path)

for subdir, _, files in os.walk(original_dicoms):
    subfolder = os.path.basename(subdir)
    r = df.index[df["Subfolder"] == subfolder].tolist()
    if r != []:
        r = r[0]
    
        filepaths_list = []
        for file in files:
            filepaths_list.append(os.path.join(subdir, file))
        filepaths_list.sort()
        
        i = 0
        for filepath in filepaths_list:
            num_string = f"{i:04d}"
            os.rename(filepath, f"{subdir}/IMG_{df['ID'][r]}_{num_string}")
            i += 1