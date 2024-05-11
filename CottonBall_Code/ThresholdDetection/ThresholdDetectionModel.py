import cv2
import csv
import numpy as np
import config
import os

def get_img_list(directory):
    filepaths = []
    for filename in os.listdir(directory):
        if filename.endswith("_no_box.png"):
            filepath = os.path.join(directory, filename)
            filepaths.append(filepath)
    return filepaths

def load_image(image_path):
    return cv2.imread(image_path, cv2.IMREAD_ANYCOLOR)

def get_mean_stdev(image):
    mean, stdev = cv2.meanStdDev(image)[0][0][0], cv2.meanStdDev(image)[1][0][0]
    return mean, stdev

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

def filter_percent(contours_dictionary_list, area_percent=100, circularity_percent=100, compactness_percent=100, solidity_percent=100, majorAxisLength_percent=100, minorAxisLength_percent=100):
    filters = [
        ("area", area_percent),
        ("circularity", circularity_percent),
        ("compactness", compactness_percent),
        ("solidity", solidity_percent),
        ("majorAxisLength", majorAxisLength_percent),
        ("minorAxisLength", minorAxisLength_percent)
    ]

    filtered = contours_dictionary_list.copy()

    for filter_key, percent in filters:
        filtered = sorted(filtered, key=lambda contour: contour[filter_key], reverse=True)
        idx = int(len(filtered) * percent / 100)
        filtered = filtered[:idx]

    return filtered


def filter_top_n(contours_dictionary_list, area_idx=-1, circularity_idx=-1, compactness_idx=-1, solidity_idx=-1, majorAxisLength_idx=-1, minorAxisLength_idx=-1):
    filters = [
        ("area", area_idx),
        ("circularity", circularity_idx),
        ("compactness", compactness_idx),
        ("solidity", solidity_idx),
        ("majorAxisLength", majorAxisLength_idx),
        ("minorAxisLength", minorAxisLength_idx)
    ]

    filtered = contours_dictionary_list.copy()

    for filter_key, idx in filters:
        idx = len(filtered) if idx == -1 else idx
        filtered = sorted(filtered, key=lambda contour: contour[filter_key], reverse=True)
        filtered = filtered[:idx]

    return filtered


def filter_value(contours_dictionary_list, area_thresh=0, circularity_thresh=0, compactness_thresh=0, solidity_thresh=0, majorAxisLength_thresh=0, minorAxisLength_thresh=0):
    filters = [
        ("area", area_thresh),
        ("circularity", circularity_thresh),
        ("compactness", compactness_thresh),
        ("solidity", solidity_thresh),
        ("majorAxisLength", majorAxisLength_thresh),
        ("minorAxisLength", minorAxisLength_thresh)
    ]

    filtered = contours_dictionary_list

    for filter_key, thresh in filters:
        filtered = [contour for contour in filtered if contour[filter_key] > thresh]
    
    return filtered


def filter_weighted(contours_dictionary_list, n=-1, area_weight=0, circularity_weight=0, compactness_weight=0, solidity_weight=0, majorAxisLength_weight=0, minorAxisLength_weight=0):
    filters = [
        ("area", area_weight),
        ("circularity", circularity_weight),
        ("compactness", compactness_weight),
        ("solidity", solidity_weight),
        ("majorAxisLength", majorAxisLength_weight),
        ("minorAxisLength", minorAxisLength_weight)
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

def display_images(image_list):
    for i, image in enumerate(image_list):
        cv2.imshow(f"Image {i+1}" + str(image), image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def preprocess(image, gauss_val, thresh_val, show_images=False): #input image  
    grayscale_image = np.copy(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    k = round(gauss_val) if round(gauss_val) % 2 == 1 else round(gauss_val)-1
    gauss_blurred_image = cv2.GaussianBlur(grayscale_image, (k,k), 0)
    _, binary_image = cv2.threshold(gauss_blurred_image, thresh_val, 255, type=cv2.THRESH_BINARY)

    if show_images:
        display_images([image, cropped_image, gauss_blurred_image, binary_image])

    return binary_image

def sort_and_display_contours(image, contours_dictionary_list, sort_key, n=5):
    sorted_contours = sorted(contours_dictionary_list, key=lambda contour: contour[sort_key], reverse=True)[:n]
    sorted_contour_list = [contour["contour"] for contour in sorted_contours]
    
    sorted_image = np.copy(image)
    cv2.drawContours(sorted_image, sorted_contour_list, -1, (255, 0, 0), 2)
    
    display_images([sorted_image])

def isolate_contours(image, preprocessed_image, forCropping=False, show_images=True): #input preprocessed image
    contours, _ = cv2.findContours(preprocessed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_dictionaryList = create_contour_dictionaryList(contours)
    if forCropping:
        filtered_dictionaryList = filter_top_n(contours_dictionaryList, 1, 1, 1)
    else:
        filtered_dictionaryList = contours_dictionaryList
        filtered_dictionaryList = filter_weighted(filtered_dictionaryList, 3, 10, 1, 1)
    #sort_and_display_contours(image, filtered_dictionaryList, 'minorAxisLength', 5) #for testing purposes
    filtered_contours = [contour["contour"] for contour in filtered_dictionaryList]
    if show_images:
        contours_image = np.copy(image)
        cv2.drawContours(contours_image, filtered_contours, -1, (255, 0, 255), 3)
        display_images([contours_image])
    return filtered_contours

def draw_boxes(image, contours, show_images=False): #input original image and filtered contours
    bboxes_image = np.copy(image)
    bboxes_list = []
    for contour in contours:
        bounding_rect = cv2.boundingRect(contour)
        bboxes_list.append(bounding_rect)
        cv2.rectangle(bboxes_image, (bounding_rect[0], bounding_rect[1]),(bounding_rect[0] + bounding_rect[2], bounding_rect[1] + bounding_rect[3]),(0, 255, 0), 3)
    corr_bboxes_image, corr_bboxes_list = bboxes_image, bboxes_list
    if show_images:
        display_images([image, bboxes_image, corr_bboxes_image])
    return corr_bboxes_list

def crop_image(image): #rewrite this in terms of preprocess, etc.
    preprocessed_image = preprocess(image, gauss_val = 1, thresh_val = 1, show_images = False)
    filtered_contours = isolate_contours(image, preprocessed_image, forCropping = True, show_images = False)
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
    bboxes_image = np.copy(image)
    bboxes_list = []
    for contour in contours:
        bounding_rect = cv2.boundingRect(contour)
        bboxes_list.append(bounding_rect)
        cv2.rectangle(bboxes_image, (bounding_rect[0], bounding_rect[1]),(bounding_rect[0] + bounding_rect[2], bounding_rect[1] + bounding_rect[3]),(0, 255, 0), 3)
    if show_images:
        display_images([bboxes_image])
    return bboxes_list

def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    x_intersection = max(x1, x2)
    y_intersection = max(y1, y2)
    w_intersection = max(0, min(x1 + w1, x2 + w2) - x_intersection)
    h_intersection = max(0, min(y1 + h1, y2 + h2) - y_intersection)
    
    area_intersection = w_intersection * h_intersection
    area_box1 = w1 * h1
    area_box2 = w2 * h2
    iou = area_intersection / (area_box1 + area_box2 - area_intersection)
    
    return iou

def calculate_iou_matrix(estimate_bboxes, gt_bboxes, output_csv=False):
    iou_matrix = []
    for estimate_bbox in estimate_bboxes:
        max_iou = 0.0
        for gt_bbox in gt_bboxes:
            iou_value = calculate_iou(estimate_bbox, gt_bbox)
            max_iou = max(max_iou, iou_value)
        iou_matrix.append(max_iou)
    accuracy = sum(iou_matrix) / len(iou_matrix)

    if output_csv:
        data_to_append = iou_matrix + [accuracy]
        csv_filename = "/Users/jayvik/Documents/GitHub/HEPIUS/CottonBalls/ThresholdDetection/output/output_csv.csv"
        with open(csv_filename, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(data_to_append)

    return iou_matrix, accuracy

if __name__ == "__main__":
    data_dir = "/Users/jayvik/Documents/GitHub/HEPIUS/CottonBalls/datasets/"
    test_list = get_img_list(data_dir)
    accuracy_list = []
    for img_path in test_list:
        show_image = False

        img = load_image(img_path)

        cropped_image, _ = crop_image(img)
        mean, stdev = get_mean_stdev(cropped_image)
        gauss_val = stdev * 0.1
        thresh_val = mean + stdev * 1
        preprocessed_image = preprocess(img, gauss_val, thresh_val, show_image)
        filtered_contours = isolate_contours(img, preprocessed_image, False, show_image)
        bboxes_list = draw_boxes(img, filtered_contours, show_image)

        real_img = load_image(img_path[:-11]+'_boxes.png')
        real_bboxes_list = find_boxes(real_img, show_image)
        
        iou_matrix, accuracy = calculate_iou_matrix(bboxes_list, real_bboxes_list, output_csv=True)
        accuracy_list.append(accuracy)
        print(img_path[59:-11] + ":", np.round(accuracy, 3), np.round(iou_matrix, 3))
    print(round(np.mean(accuracy_list), 3))
