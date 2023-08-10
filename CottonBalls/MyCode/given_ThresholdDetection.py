"""
Before implementing a deep neural network, we should set a baseline to beat. In this file, we use the fact that
cotton is much brighter in the image than the rest of the tissue.

Next step will be to use template matching to find the cotton ball.

Finally, a neural network can be implemented, starting with a simple CNN and moving on to faster RCNN.

Here:
- threshold image
- get largest object
- find threshold in a few ways: (1) use FWHM; (2) calculate average pixel value in box of each image and use the
95th percentile of these values
"""
import numpy as np
import cv2
import skimage.filters as skimg
import glob
import os
import config
import pandas as pd
import matplotlib.pyplot as plt


def visualize_boxes(img, boxA, boxB):
    x, y, w, h = boxA
    h = int(h)
    res = cv2.rectangle(img.copy(), (x, y), (x + w, y + h), (0, 255, 0), 2)
    x, y, w, h = boxB
    h = int(h)
    res = cv2.rectangle(res, (x, y), (x + w, y + h), (0, 0, 255), 2)
    plt.figure(1)
    plt.imshow(res)
    plt.show()


def threshold_img(img, threshold=100):
    """
      1) get fwhm value and threshold
      2) use object detection from opencv and take the largest object found
      3) capture left and rightmost, top and bottom-most pixel locations and form a box
      4) return locations of this box
      :param img: pixel_array of input image
      :param threshold: pixel value to mark end of possible cotton
      :return: (x, y), width, height of box around predicted cotton ball
      """
    img_thresh = np.asarray(np.where(img > threshold, 1, 0), dtype=np.uint8)
    img_open = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, np.ones((10, 10), np.uint8))
    comp = cv2.connectedComponentsWithStats(img_open, connectivity=8)
    bb = comp[2][np.argmax(comp[2][1:, -1]) + 1, :-1]

    return bb


# def calculate_mean_gt_pixel_val():
#     """
#     - go through each ground truth image and calculate the average pixel value within the bounding box
#     - keep the avg pixel values in an array to then get the 5th and 95th percentiles and other stats
#     - use 95th percentile as threshold value
#     """
#     # get all dcm files annotated with ground truth
#     gt_dataframe = pd.read_csv(os.path.join(config.dcm_date_path, 'gt_annotations.csv'), header=None)
#     orig_files = sorted(glob.glob(os.path.join(config.input_path, '*.png')))
#     avg_pix = []
#     for fname in orig_files:
#         if os.path.splitext(os.path.basename(fname))[0] in gt_dataframe[0].to_numpy():
#             bb = gt_dataframe[gt_dataframe[0] == os.path.splitext(os.path.basename(fname))[0]].to_numpy()[0][2:]
#             if np.sum(bb) == 0:
#                 continue
#             img_arr = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
#             cropped = img_arr[bb[0]:bb[0] + bb[2], bb[1]:bb[1] + bb[3]]
#             avg_pix.append(np.mean(cropped))

#     np.savetxt('avg_pix.csv', np.asarray(avg_pix), delimiter=',')
#     return avg_pix


# def analyze_avg_pix(avg_pix, percentile=95):
#     nan_array = np.isnan(avg_pix)
#     not_nan_array = ~ nan_array
#     avg_pix_nonan = avg_pix[not_nan_array]
#     #
#     # print(np.mean(avg_pix_nonan))
#     # print(np.percentile(avg_pix_nonan, 95))
#     # print(np.percentile(avg_pix_nonan, 5))

#     return np.percentile(avg_pix_nonan, percentile)


# def bb_intersection_over_union(boxA, boxB):
#     # determine the (x, y)-coordinates of the intersection rectangle
#     boxA = [boxA[0], boxA[1], boxA[0] + boxA[2], boxA[1] + boxA[3]]
#     boxB = [boxB[0], boxB[1], boxB[0] + boxB[2], boxB[1] + boxB[3]]
#     xA = max(boxA[0], boxB[0])
#     yA = max(boxA[1], boxB[1])
#     xB = min(boxA[2], boxB[2])
#     yB = min(boxA[3], boxB[3])
#     # compute the area of intersection rectangle
#     interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
#     # compute the area of both the prediction and ground-truth
#     # rectangles
#     boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
#     boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
#     # compute the intersection over union by taking the intersection
#     # area and dividing it by the sum of prediction + ground-truth
#     # areas - the interesection area
#     iou = interArea / float(boxAArea + boxBArea - interArea)
#     # return the intersection over union value
#     return iou


# def get_threshold_ious():
#     # threshold_percentile = analyze_avg_pix(np.genfromtxt('avg_pix.csv', delimiter=','), percentile=95)  # percentile
#     img_fnames = sorted(glob.glob(os.path.join(config.input_path, '*.png')))
#     all_ious = []
#     sizes = []
#     gt_dataframe = pd.read_csv(os.path.join(config.dcm_date_path, 'gt_annotations.csv'), header=None)
#     for fname in img_fnames:
#         basename = os.path.splitext(os.path.basename(fname))[0]
#         if basename not in gt_dataframe[0].to_numpy():
#             continue
#         # read image and get predicted bounding box
#         img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
#         # threshold = skimg.threshold_otsu(img)  # otsu threshold
#         threshold = np.max(img) / 2  # fwhm
#         # threshold = threshold_percentile
#         bb_pred = threshold_img(img, int(threshold))
#         # get ground truth bounding box
#         bb_gt = gt_dataframe[gt_dataframe[0] == basename].to_numpy()[0][2:]
#         if len(bb_gt) < 4:
#             bb_gt = np.append(bb_gt, [0])

#         # visualize_boxes(img, bb_pred, bb_gt)
#         iou = bb_intersection_over_union(bb_pred, bb_gt)
#         if iou > 0:
#             all_ious.append(iou)
#             sizes.append(gt_dataframe[gt_dataframe[0] == basename].to_numpy()[0][1])

#     all_ious = np.asarray(all_ious)
#     sizes = np.asarray(sizes)

#     tp = len(all_ious[all_ious > 0.5][sizes[all_ious > 0.5] != '0mm'])
#     fp = len(all_ious[all_ious < 0.5][sizes[all_ious < 0.5] == '0mm'])
#     tn = len(all_ious[all_ious > 0.5][sizes[all_ious > 0.5] == '0mm'])
#     fn = len(all_ious[all_ious < 0.5][sizes[all_ious < 0.5] != '0mm'])

#     sensitivity = tp / (tp + fn)
#     # specificity = tn / (tn + fp)
#     accuracy = len(all_ious[all_ious > 0.5]) / len(all_ious)

#     print('mean', np.mean(all_ious))
#     print('median', np.median(all_ious))
#     print('max', np.max(all_ious))
#     print('min', np.min(all_ious))
#     print('sensitivity', sensitivity)
#     # print('specificity', specificity)
#     print('accuracy', accuracy)



if __name__ == "__main__":
    # avg_px = calculate_mean_gt_pixel_val()
    
    ###get_threshold_ious()

    # find average pixel value of ground truth cotton
    # avg_pix = calculate_mean_gt_pixel_val()
    # avg_pix = np.genfromtxt('avg_pix.csv', delimiter=',')
    # analyze_avg_pix(avg_pix)
    img = cv2.imread(config.input_path, cv2.IMREAD_GRAYSCALE)
    bb = threshold_img(img)
    visualize_boxes(img, bb, bb)