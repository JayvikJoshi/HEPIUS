import cv2
import numpy as np
import pydicom as pdcm
import matplotlib.pyplot as plt
import glob
import os
from sahi.utils.coco import Coco, CocoCategory, CocoImage, CocoAnnotation
from sahi.utils.file import save_json
import config

coco_id_dict = {'0mm': 0,
                '1mm': 1,
                '2mm': 2,
                '3mm': 3,
                '5mm': 4,
                '10mm': 5,
                '15mm': 6,
                '20mm': 7}


def create_cotton_size_dictionary(fname, size_dict):
    """
    take a filename, get its basename, make a dictionary of how many of each IDs we should have
    """
    basename_split = os.path.basename(fname).split('-')[0].split('_')
    basename_split = list(filter(None, basename_split))
    for i in range(0, len(basename_split), 2):
        num_curr_size = basename_split[i]
        curr_size = basename_split[i + 1].lower()
        if curr_size in coco_id_dict.keys():
            curr_id = coco_id_dict[curr_size.lower()]
            size_dict[curr_id] = int(num_curr_size)

    return size_dict


def get_curr_id(size_dict):
    """
    return the largest size that still has a counter number greater than 0. update the dictionary by reducing
    that size's counter number.
    :param curr_size_dictionary:
    :return:
    """
    curr_id = 0
    for key in reversed(range(8)):
        if size_dict[key] > 0:
            curr_id = key
            size_dict[key] -= 1
            break
    return curr_id, size_dict


def is_center_in_rect(center, rect):
    """
    center: (x1, y1)
    rect: (x2, y2, w, h)
    goal: x2 < x1 < x2 + w and y2 < y1 < y2 + h
    """
    return rect[0] < center[0] < rect[0] + rect[2] and rect[1] < center[1] < rect[1] + rect[3]


if __name__ == '__main__':
    # Load the image
    gt_folder = config.gt_dcm_folder
    display_imgs = False

    all_dcms = sorted(glob.glob(os.path.join(gt_folder, '*.dcm')))

    # make a Coco dataset for use with YOLOv8
    coco = Coco()
    coco.add_category(CocoCategory(id=0, name='0mm'))
    coco.add_category(CocoCategory(id=1, name='1mm'))
    coco.add_category(CocoCategory(id=2, name='2mm'))
    coco.add_category(CocoCategory(id=3, name='3mm'))
    coco.add_category(CocoCategory(id=4, name='5mm'))
    coco.add_category(CocoCategory(id=5, name='10mm'))
    coco.add_category(CocoCategory(id=6, name='15mm'))
    coco.add_category(CocoCategory(id=7, name='20mm'))

    for fname in all_dcms:
        print(fname)

        # use this to test a single file
        # only mistake currently is that boxes that are their own rectangles are not included in the larger rectangle contours
        # if '3_10MM-dicom-00001_00006.dcm' not in fname:
        #     continue
        # 3_10MM-dicom-00001_00006.dcm <- gotta decide whether we want to handle this
        # 3_10MM-dicom-00001_00042.dcm <- gotta decide whether we want to handle this
        # 4_10MM-dicom-00001_00001.dcm <- picks wrong end piece of contour
        # 4_10MM-dicom-00001_00004.dcm <- picks wrong end piece of contour
        # 4_10MM-dicom-00001_00009.dcm <- picks wrong end piece of contour (also many of these others)
        # 4_10MM-dicom-00001_00011.dcm <- picks wrong end piece of contour

        # load dicom
        dcm = pdcm.dcmread(fname)
        img = dcm.pixel_array
        if dcm.PhotometricInterpretation != 'RGB':
            img = pdcm.pixel_data_handlers.util.convert_color_space(img, dcm.PhotometricInterpretation,
                                                                    'RGB',
                                                                    per_frame=True)

        # make this a coco image
        coco_image = CocoImage(file_name=os.path.basename(fname.replace('dcm', 'png')), height=config.og_img_height, width=config.og_img_width)
        # make a dictionary of what cotton ball sizes we're looking for
        curr_size_dictionary = create_cotton_size_dictionary(fname, size_dict={i: 0 for i in range(8)})

        # capture yellow boxes
        mask = np.asarray(
            np.where((img[:, :, 0] > 127) & (img[:, :, 1] > 127) & (img[:, :, 2] <= 127),
                     255, 0), dtype=np.uint8)

        # Find contours of the objects in the binary image
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # if no cotton balls in image then just add 0s and image to dataset and continue
        if len(contours) == 0:
            coco_image.add_annotation(
                CocoAnnotation(
                    bbox=[0.0, 0.0, 0.0, 0.0],
                    category_id=0,
                    category_name=0
                )
            )
            coco.add_image(coco_image)
            continue

        # figure out which contour is just surrounding multiple other contours
        num_occurrences_parents = np.asarray([hierarchy[0][i][3]
                                              for i in range(len(hierarchy[0])) if hierarchy[0][i][3] > -1])
        # a parent is too big if more than two children have it as its parent
        parents_too_big = []

        # error: big parent has two children but not 3
        if len(num_occurrences_parents) > 0:
            bins = np.bincount(num_occurrences_parents)
            # either bins > 2 or (bins == 2 and hierarchy[0][-1][3] == bins and len(hierarchy[0] > 2))
            parents_too_big = [np.logical_or(bins[i] > 2, np.logical_and(bins[i] == 2, len(hierarchy[0] > 2)))
                               for i in range(len(bins))]

        # get the centers and rectangles of each contour
        all_centers = []
        all_centers_plus = []
        rect_dim = []
        for i, cnt in enumerate(contours):
            # skip if this is the large parent
            if i < len(parents_too_big) and parents_too_big[i]:
                continue

            # check if the center of a rectangle only falls within one contour/rectangle
            # if the center is in multiple rectangles, then it must be part of an overlap, so we don't count it
            # include a few other points around the center just in case
            x, y, w, h = cv2.boundingRect(cnt)
            # calculate center as moment
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                # skip repeating centers, eg., if a contour counts the inside of a box and outside of a box separately
                if np.any([item in all_centers for item in [[cx, cy], [cx + 1, cy], [cx, cy + 1], [cx + 1, cy + 1],
                                                            [cx - 1, cy], [cx, cy - 1], [cx - 1, cy - 1],
                                                            [cx - 1, cy + 1], [cx + 1, cy - 1]]]):
                    continue
                all_centers.append([cx, cy])
                # make a list of 5 points to check if they are all within the bounds of another rectangle
                all_centers_plus.append([[cx, cy], [cx + w//4, cy], [cx - w//4, cy],
                                         [cx, cy + h//4], [cx, cy - h//4]])
                rect_dim.append((x, y, w, h))

        print(len(all_centers))

        # go through each center and surrounding points, make sure they're all in only 1 rectangle, then
        for i in range(len(all_centers)):
            inside_count = 0
            for j in range(len(rect_dim)):
                inside_count += np.alltrue([is_center_in_rect(all_centers_plus[i][k], rect_dim[j])
                                            for k in range(len(all_centers_plus[i]))])
            # if center point only in one rectangle, then add this rectangle as an annotation to the COCO dataset
            if inside_count <= 1 or inside_count == len(contours):
                x, y, w, h = rect_dim[i]

                curr_id, curr_size_dictionary = get_curr_id(curr_size_dictionary)

                coco_image.add_annotation(
                    CocoAnnotation(
                        bbox=[x, y, w, h],
                        category_id=curr_id,
                        category_name=curr_id
                    )
                )

                # Draw a rectangle around the contour
                if display_imgs:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # add the image with its annotations to our overall coco dataset
        coco.add_image(coco_image)

        # Display the result
        if display_imgs:
            cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imshow('result', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    # save our dataset
    if config.save_down:
        save_json(data=coco.json, save_path=config.ground_truth_file)
