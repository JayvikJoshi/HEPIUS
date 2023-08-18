import cv2
import numpy as np
import config
import os
#from skimage.feature import greycomatrix, greycoprops

def get_img_list(directory):
    filepaths = []
    for filename in os.listdir(directory):
        if filename.endswith("no_box.png"):
            filepath = os.path.join(directory, filename)
            filepaths.append(filepath)
    return filepaths

def load_image(image_path):
    return cv2.imread(image_path, cv2.IMREAD_ANYCOLOR)


def get_mean_stdev(image):
    mean, stdev = cv2.meanStdDev(image)[0][0][0], cv2.meanStdDev(image)[1][0][0]
    return mean, stdev

def apply_gaussian_blur(image, kernel_size=(5,5)):
    return cv2.GaussianBlur(image, kernel_size, 0)

def perform_thresholding(image, threshold_value, max_value=255, threshold_type=cv2.THRESH_BINARY):
    _, binary_threshold = cv2.threshold(image, threshold_value, max_value, threshold_type)
    return binary_threshold

def find_contours(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def create_contour_dictionaryList(contours):
    contours_dictionary_list = []

    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter != 0 else 0
        compactness = area / (perimeter ** 2) if perimeter != 0 else 0
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area != 0 else 0  # Handle division by zero
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            majorAxisLength, minorAxisLength = ellipse[1]
        else:
            majorAxisLength, minorAxisLength = 0, 0
        
        contour_dictionary = {
            "contour": contour,
            "area": area,
            "circularity": circularity,
            "compactness": compactness,
            "solidity": solidity,
            "majorAxisLength": majorAxisLength,
            "minorAxisLength": minorAxisLength
        }
        contours_dictionary_list.append(contour_dictionary)

    return contours_dictionary_list


def filter_percent(contours_dictionary_list, area_percent=100, circularity_percent=100, compactness_percent=100):
    filters = [
        ("area", area_percent),
        ("circularity", circularity_percent),
        ("compactness", compactness_percent)
    ]

    filtered = contours_dictionary_list.copy()

    for filter_key, percent in filters:
        filtered = sorted(filtered, key=lambda contour: contour[filter_key], reverse=True)
        idx = int(len(filtered) * percent / 100)
        filtered = filtered[:idx]

    return filtered

def filter_top_n(contours_dictionary_list, area_idx=-1, circularity_idx = -1, compactness_idx=-1):
    filters = [
        ("area", area_idx),
        ("circularity", circularity_idx),
        ("compactness", compactness_idx)
    ]

    filtered = contours_dictionary_list.copy()

    for filter_key, idx in filters:
        idx = len(filtered) if idx == -1 else idx
        filtered = sorted(filtered, key=lambda contour: contour[filter_key], reverse=True)
        filtered = filtered[:idx]

    return filtered

def filter_value(contours_dictionary_list, area_thresh=0, circularity_thresh=0, compactness_thresh=0):
    filters = [
        ("area", area_thresh),
        ("circularity", circularity_thresh),
        ("compactness", compactness_thresh)
    ]

    filtered = contours_dictionary_list

    for filter_key, thresh in filters:
        filtered = [contour for contour in filtered if contour[filter_key] > thresh]
    
    return filtered


def filter_weighted(contours_dictionary_list, n=-1, area_weight=0, circularity_weight=0, compactness_weight=0):
    filters = [
        ("area", area_weight),
        ("circularity", circularity_weight),
        ("compactness", compactness_weight)
    ]
    
    if n == -1:
        n = len(contours_dictionary_list)
    
    for contour in contours_dictionary_list:
        contour["weighted_score"] = 0
        for key, weight in filters:
            key_min = min(contours_dictionary_list, key=lambda x: x[key])[key]
            key_max = max(contours_dictionary_list, key=lambda x: x[key])[key]
            contour_values = {
                key: (contour[key] - key_min) / (key_max - key_min) if key_max != key_min else 0
            }
            contour["weighted_score"] += contour_values[key] * weight
    
    weight_filtered = sorted(contours_dictionary_list, key=lambda contour: contour["weighted_score"], reverse=True)[:n]
    return weight_filtered

def display_contours(image, contours):
    contours_image = np.copy(image)
    cv2.drawContours(contours_image, contours, -1, (255, 0, 255), 3)
    return contours_image

def display_images(image_list):
    for i, image in enumerate(image_list):
        cv2.imshow(f"Image {i+1}" + str(image), image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def correct_bboxes(image, bboxes_list):
    _, crop_dimensions = crop_image(image)
    corr_bboxes_list = []
    for bbox_value in bboxes_list:
        bbox_value_list, crop_dimensions_list = list(bbox_value), list(crop_dimensions)
        corr_bbox_value = tuple([x + y for x, y in zip(bbox_value_list, crop_dimensions_list)])
        corr_bboxes_list.append(corr_bbox_value)

    corr_bboxes_image = np.copy(image)
    for corr_bbox_value in corr_bboxes_list:
        x_min, y_min, width, height = corr_bbox_value
        cv2.rectangle(corr_bboxes_image, (x_min, y_min), (x_min + width, y_min + height), (0, 255, 0), 2)

    return corr_bboxes_image, corr_bboxes_list

def preprocess(image, gauss_val, thresh_val, show_images=False): #input image
    grayscale_image = np.copy(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    k = round(gauss_val) if round(gauss_val) % 2 == 1 else round(gauss_val)-1
    gauss_blurred_image = apply_gaussian_blur(grayscale_image, (k,k))
    binary_image = perform_thresholding(gauss_blurred_image, thresh_val)

    if show_images:
        display_images([image, cropped_image, gauss_blurred_image, binary_image])

    return binary_image

def isolate_contours(image, preprocessed_image, n, area_weight, circ_weight, comp_weight, show_images=True): #input preprocessed image
    contours, _ = cv2.findContours(preprocessed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_dictionaryList = create_contour_dictionaryList(contours)
    filtered_dictionaryList = filter_weighted(contours_dictionaryList, n, area_weight, circ_weight, comp_weight)
    filtered_contours = [contour["contour"] for contour in filtered_dictionaryList]
    if show_images:
        display_images([display_contours(image, filtered_contours)])
    return filtered_contours

def draw_boxes(image, contours, show_images=False): #input original image and filtered contours
    bboxes_image = np.copy(image)
    bboxes_list = []
    for contour in contours:
        bounding_rect = cv2.boundingRect(contour)
        bboxes_list.append(bounding_rect)
        cv2.rectangle(bboxes_image, (bounding_rect[0], bounding_rect[1]),(bounding_rect[0] + bounding_rect[2], bounding_rect[1] + bounding_rect[3]),(0, 255, 0), 3)
    corr_bboxes_image, corr_bboxes_list = bboxes_image, bboxes_list #correct_bboxes(image, bboxes_list)
    if show_images:
        display_images([image, bboxes_image, corr_bboxes_image])
    return corr_bboxes_list

def crop_image(image): #rewrite this in terms of preprocess, etc.
    preprocessed_image = preprocess(image, gauss_val = 1, thresh_val = 1, show_images = False)
    filtered_contours = isolate_contours(image, preprocessed_image, 1, 100, 0, 0, False)
    largest_contour = filtered_contours[0]
    bounding_rect = cv2.boundingRect(largest_contour)
    cropped_image = image[bounding_rect[1]:bounding_rect[1] + bounding_rect[3], bounding_rect[0]:bounding_rect[0] + bounding_rect[2]]
    crop_dimensions = (bounding_rect[0], bounding_rect[1], 0, 0)
    return cropped_image, crop_dimensions


def find_boxes(image, show_images=True):
    lower_yellow = np.array([0, 100, 100])
    upper_yellow = np.array([1, 255, 255])
    yellow_mask = cv2.inRange(image, lower_yellow, upper_yellow)
    contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes_image = np.copy(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
    bboxes_list = []
    for contour in contours:
        bounding_rect = cv2.boundingRect(contour)
        bboxes_list.append(bounding_rect)
        cv2.rectangle(bboxes_image, (bounding_rect[0], bounding_rect[1]),(bounding_rect[0] + bounding_rect[2], bounding_rect[1] + bounding_rect[3]),(0, 255, 0), 3)
    if show_images:
        display_images([bboxes_image])
    return bboxes_list

def calculate_iou(box1, box2):
    x1_real, y1_real, w1_real, h1_real = box1
    x2_real, y2_real, w2_real, h2_real = x1_real + w1_real, y1_real + h1_real, w1_real, h1_real
    x1_est, y1_est, w1_est, h1_est = box2
    x2_est, y2_est, w2_est, h2_est = x1_est + w1_est, y1_est + h1_est, w1_est, h1_est

    x_left = max(x1_real, x1_est)
    y_top = max(y1_real, y1_est)
    x_right = min(x2_real, x2_est)
    y_bottom = min(y2_real, y2_est)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = w1_real * h1_real
    box2_area = w1_est * h1_est
    union_area = box1_area + box2_area - intersection_area

    iou = intersection_area / union_area
    return iou

    

if __name__ == "__main__":
    img = load_image("/Users/jayvik/Documents/GitHub/HEPIUS/CottonBalls/datasets/3_5mm_no_box.png")
    real_img = load_image("/Users/jayvik/Documents/GitHub/HEPIUS/CottonBalls/datasets/3_5mm_boxes.png")
    

    cropped_image, _ = crop_image(img)
    mean, stdev = get_mean_stdev(cropped_image)
    gauss_val = stdev * 0.1
    thresh_val = mean + stdev * 0.8

    preprocessed_image = preprocess(img, gauss_val, thresh_val, True)
    filtered_contours = isolate_contours(img, preprocessed_image, 2, 100, 2, 2, True)
    bboxes_list = draw_boxes(img, filtered_contours, True)
    real_bboxes_list = find_boxes(real_img, True)



    #     # 2_3mm_no_boxes -> threshold at 175, size (0.001, 0.1), circularity (0.1, 1), compactness (0.1, 1)
    #     # 2_20mm_no_boxes -> threshoold at 100, size (0.03, 1), circularity (0.03, 1), compactness (0.03, 1)