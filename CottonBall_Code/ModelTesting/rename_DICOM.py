#goal: take a folder of DICOM files and match them to .png files by name, then rename the DICOM files to match the .png files
#input: folder of DICOM files, folder of .png files
#output: DICOM files renamed to match .png files

import os
import pydicom
import shutil

#path to DICOM files
dicom_path = "/Users/jayvik/Documents/GitHub/HEPIUS/CottonBall_Data/