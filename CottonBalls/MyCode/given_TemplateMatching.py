"""
Find a few different cotton ball templates. Then perform template matching as in
"""

import numpy as np
import argparse
import glob
import cv2
import os
import matplotlib.pyplot as plt
import pandas as pd
import config
from threshold_detection_model import bb_intersection_over_union, visualize_boxes


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized


def translate(image, x, y):
    # Define the translation matrix and perform the translation
    M = np.float32([[1, 0, x], [0, 1, y]])
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    # Return the translated image
    return shifted


def rotate(image, angle, center = None, scale = 1.0):
    # Grab the dimensions of the image
    (h, w) = image.shape[:2]

    # If the center is None, initialize it as the center of
    # the image
    if center is None:
        center = (w / 2, h / 2)

    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    # Return the rotated image
    return rotated


# method for template matching with resize. apply to each transverse cine image
def template_match(test_img, img_size, template, visualize):
    found = None
    img_2d = np.reshape(test_img, img_size)
    img_2d = np.asarray(img_2d, dtype=np.uint8)
    template = np.asarray(template.copy(), dtype=np.uint8)
    (tH, tW) = template.shape[:2]
    for scale in np.linspace(0.2, 1.0, 20)[::-1]:
        # resize the image according to the scale, and keep track
        # of the ratio of the resizing
        resized = resize(img_2d, width=int(img_2d.shape[1] * scale))
        r = img_2d.shape[1] / float(resized.shape[1])

        # if the resized image is smaller than the template, then break
        # from the loop
        if resized.shape[0] < tH or resized.shape[1] < tW:
            break

        # detect edges in the resized, grayscale image and apply template
        # matching to find the template in the image
        result = cv2.matchTemplate(resized, template, cv2.TM_CCOEFF)
        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

        # if we have found a new maximum correlation value, then ipdate
        # the bookkeeping variable
        if found is None or maxVal > found[0]:
            found = (maxVal, maxLoc, r)
        # unpack the bookkeeping variable and compute the (x, y) coordinates

    # of the bounding box based on the resized ratio
    (_, maxLoc, r) = found
    (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
    (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))

    # draw a bounding box around the detected result and display the image
    rect_img = cv2.rectangle(img_2d, (startX, startY), (endX, endY), (0, 100, 100), 2)
    if visualize:
        plt.figure()
        plt.imshow(rect_img, cmap='gray')

    # return new coordinates for image to crop to
    return max(0, startX - 25), max(startY - 25, 0), min(img_size[0], endX + 25), min(img_size[1], endY + 25)


if __name__ == '__main__':
    template = cv2.imread(os.path.join(config.dcm_date_path, 'templates', 'template4.png'), cv2.IMREAD_GRAYSCALE)
    input_img_fnames = sorted(glob.glob(os.path.join(config.input_path, '*.png')))
    gt_dataframe = pd.read_csv(os.path.join(config.dcm_date_path, 'gt_annotations.csv'), header=None)
    all_ious = []
    sizes = []
    for fname in input_img_fnames:
        basename = os.path.splitext(os.path.basename(fname))[0]
        if basename not in gt_dataframe[0].to_numpy():
            continue
        img = np.asarray(cv2.imread(fname, cv2.IMREAD_UNCHANGED), dtype=np.uint8)
        x1, y1, x2, y2 = template_match(img, np.shape(img), template, False)
        bb_pred = np.array([x1, y1, np.abs(y2 - y1), np.abs(x2 - x1)])

        bb_gt = gt_dataframe[gt_dataframe[0] == basename].to_numpy()[0][2:]

        iou = bb_intersection_over_union(bb_pred, bb_gt)
        # if iou > 0:
        all_ious.append(iou)
        sizes.append(gt_dataframe[gt_dataframe[0] == basename].to_numpy()[0][1])

    all_ious = np.asarray(all_ious)
    sizes = np.asarray(sizes)

    tp = len(all_ious[all_ious > 0.5][sizes[all_ious > 0.5] != '0mm'])
    fp = len(all_ious[all_ious < 0.5][sizes[all_ious < 0.5] == '0mm'])
    tn = len(all_ious[all_ious > 0.5][sizes[all_ious > 0.5] == '0mm'])
    fn = len(all_ious[all_ious < 0.5][sizes[all_ious < 0.5] != '0mm'])

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    accuracy = len(all_ious[all_ious > 0.5]) / len(all_ious)

    print('mean', np.mean(all_ious))
    print('median', np.median(all_ious))
    print('max', np.max(all_ious))
    print('min', np.min(all_ious))
    print('sensitivity', sensitivity)
    print('specificity', specificity)
    print('accuracy', accuracy)