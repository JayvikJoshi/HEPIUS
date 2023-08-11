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
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

def crop_image(image):
    grayscale_image = image
    threshold = 1
    binary_image = cv2.threshold(grayscale_image, threshold, 255, cv2.THRESH_BINARY)[1]

    # Find the contours in the binary image.
    contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour, which should be the ultrasound image.
    largest_contour = None
    largest_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > largest_area:
            largest_contour = contour
            largest_area = area

    # Crop the image to the largest contour.
    cropped_image = cv2.drawContours(image, [largest_contour], -1, (0, 255, 0), 3)

    # Get the bounding rectangle of the contour.
    bounding_rect = cv2.boundingRect(largest_contour)

    # Crop the image to the bounding rectangle.
    cropped_image = cropped_image[bounding_rect[1]:bounding_rect[1] + bounding_rect[3], bounding_rect[0]:bounding_rect[0] + bounding_rect[2]]
    return cropped_image

def get_mean_stdev(image):
    mean, stdev = cv2.meanStdDev(image)[0][0][0], cv2.meanStdDev(image)[1][0][0]
    return mean, stdev

def apply_gaussian_blur(image, kernel_size=(5,5)):
    return cv2.GaussianBlur(image, kernel_size, 0)

def perform_thresholding(image, threshold_value, max_value=255, threshold_type=cv2.THRESH_BINARY):
    _, binary_threshold = cv2.threshold(image, threshold_value, max_value, threshold_type)
    return binary_threshold

def find_contours(binary_image):
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def create_contour_dictionary_list(contours):
    contours_dictionary_list = []
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter != 0 else -1
        compactness = area / (perimeter * perimeter) if perimeter != 0 else -1
        if circularity != -1 and compactness != -1:
            contour_dictionary = {}
            contour_dictionary["contour"] = contour
            contour_dictionary["area"] = area
            contour_dictionary["circularity"] = circularity
            contour_dictionary["compactness"] = compactness
            contours_dictionary_list.append(contour_dictionary)
    return contours_dictionary_list

def filter_contours_percent(contours_dictionary_list, area_percent=100, circularity_percent=100, compactness_percent=100):
    #area filter
    area_sorted = sorted(contours_dictionary_list, key=lambda contour: contour["area"], reverse=True)
    area_idx = int(len(area_sorted) * area_percent / 100)
    area_filtered = area_sorted[:area_idx]
    #circ filter
    circularity_sorted = sorted(area_filtered, key=lambda contour: contour["circularity"], reverse=True)
    circularity_idx = int(len(circularity_sorted) * circularity_percent / 100)
    circularity_filtered = circularity_sorted[:circularity_idx]
    #comp filter
    compactness_sorted = sorted(circularity_filtered, key=lambda contour: contour["compactness"], reverse=True)
    compactness_idx = int(len(compactness_sorted) * compactness_percent / 100)
    compactness_filtered = compactness_sorted[:compactness_idx]

    filtered = compactness_filtered
    return filtered

def filter_contours_number(contours_dictionary_list, area_idx=-1, circularity_idx = -1, compactness_idx=-1):
    
    area_idx = len(contours_dictionary_list) if area_idx == -1 else area_idx
    circularity_idx = len(contours_dictionary_list) if circularity_idx == -1 else circularity_idx
    compactness_idx = len(contours_dictionary_list) if compactness_idx == -1 else compactness_idx
    #area filter
    area_sorted = sorted(contours_dictionary_list, key=lambda contour: contour["area"], reverse=True)
    area_filtered = area_sorted[:area_idx]
    #circ filter
    circularity_sorted = sorted(area_filtered, key=lambda contour: contour["circularity"], reverse=True)
    circularity_filtered = circularity_sorted[:circularity_idx]
    #comp filter
    compactness_sorted = sorted(circularity_filtered, key=lambda contour: contour["compactness"], reverse=True)
    compactness_filtered = compactness_sorted[:compactness_idx]
    
    filtered = compactness_filtered
    return filtered

def filter_contours_thresh(contours_dictionary_list, area_thresh = 0, circularity_thresh = 0, compactness_thresh = 0):
    #area filter
    area_sorted = sorted(contours_dictionary_list, key=lambda contour: contour["area"], reverse=True)
    for index, contour in enumerate(contours_dictionary_list):
        area_idx = index
        if contour["area"] <= area_thresh:
            break
    area_filtered = area_sorted[:area_idx]
    #circ filter
    circularity_sorted = sorted(area_filtered, key=lambda contour: contour["circularity"], reverse=True)
    for index, contour in enumerate(area_filtered):
        circularity_idx = index
        if contour["circularity"] <= circularity_thresh:
            break
    circularity_filtered = circularity_sorted[:circularity_idx]
    #comp filter
    compactness_sorted = sorted(circularity_filtered, key=lambda contour: contour["compactness"], reverse=True)
    for index, contour in enumerate(circularity_filtered):
        compactness_idx = index
        if contour["circularity"] <= compactness_thresh:
            break 
    compactness_filtered = compactness_sorted[:compactness_idx]    
    filtered = compactness_filtered

    return filtered

def filter_contours_weight(contours_dictionary_list, n=-1, area_weight=0, circularity_weight=0, compactness_weight=0):
    n = len(contours_dictionary_list) if n == -1 else n

    
    area_list = [contour["area"] for contour in contours_dictionary_list]
    circularity_list = [contour["circularity"] for contour in contours_dictionary_list]
    compactness_list = [contour["compactness"] for contour in contours_dictionary_list]
    
    norm_contours_dictionary_list = []
    for contour in contours_dictionary_list:
        norm_contour_dictionary = {}
        norm_contour_dictionary["contour"] = contour["contour"]
        norm_contour_dictionary["area"] = (contour["area"] - min(area_list)) / (max(area_list) - min(area_list))
        norm_contour_dictionary["circularity"] = (contour["circularity"] - min(circularity_list)) / (max(circularity_list) - min(circularity_list))
        norm_contour_dictionary["compactness"] = (contour["compactness"] - min(compactness_list)) / (max(compactness_list) - min(compactness_list))
        norm_contour_dictionary["weighted_score"] = contour["area"] * area_weight + contour["circularity"] * circularity_weight + contour["compactness"] * compactness_weight
        norm_contours_dictionary_list.append(norm_contour_dictionary)

    weight_sorted = sorted(norm_contours_dictionary_list, key=lambda contour: contour["weighted_score"], reverse=True)
    
    for contour in weight_sorted:
        print(contour["area"], contour["circularity"], contour["compactness"])
    
    weight_filtered = weight_sorted[:n]
    return weight_filtered

def visualize_contours(image, contours):
    image_with_contours = np.copy(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))
    cv2.drawContours(image_with_contours, contours, -1, (255, 0, 255), 3)
    return image_with_contours

def display_images_opencv(images):
    for i, image in enumerate(images):
        cv2.imshow(f"Image {i+1}" + str(image), image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def hough_circle_transform(image, dp, min_dist, param1, param2, min_radius, max_radius):
    blurred_image = cv2.GaussianBlur(image, (9, 9), 2)
    circles = cv2.HoughCircles(blurred_image, cv2.HOUGH_GRADIENT, dp, min_dist, param1=param1, param2=param2, minRadius=min_radius, maxRadius=max_radius)
    return circles



if __name__ == "__main__":
    img_list = get_img_list(config.input_path)
    for img in img_list:
        output_list = []
        img = load_image(img)
        image = crop_image(img)
        mean, stdev = get_mean_stdev(image)

        i = stdev * 0.1
        k = round(i) if round(i) % 2 == 1 else round(i)-1
        p = 0.8
        gauss = apply_gaussian_blur(image, (k,k))
        bin = perform_thresholding(gauss, mean+stdev*p)
        
        contours = find_contours(bin)
        contour_image = visualize_contours(image, contours)
        contours_dictionary_list = create_contour_dictionary_list(contours)
        filtered = filter_contours_weight(contours_dictionary_list, 2, 1, 1, 1)
        
        filtered_contours = [contour["contour"] for contour in filtered]
        contour_filtered_image = visualize_contours(image, filtered_contours)

        # output_list.append(img) #original
        # output_list.append(image) #cropped
        # output_list.append(gauss) #gauss blurred
        # output_list.append(bin) #binarized
        # output_list.append(contour_image) #all contours
        output_list.append(contour_filtered_image) #filtered contours
        display_images_opencv(output_list)

        # dp = 1  # Inverse ratio of the accumulator resolution to the image resolution (1 means the same resolution)
        # min_dist = 50  # Minimum distance between the centers of detected circles
        # param1 = 80  # Upper threshold for the internal Canny edge detector
        # param2 = 30  # Threshold for center detection.
        # min_radius = 10  # Minimum circle radius
        # max_radius = 100  # Maximum circle radius

        # circles = hough_circle_transform(img, dp, min_dist, param1, param2, min_radius, max_radius)
        # if circles is not None:
        #     circles = np.uint16(np.around(circles))
        #     for circle in circles[0, :]:
        #         center = (circle[0], circle[1])
        #         radius = circle[2]
        #         # Draw the circle on the original image
        #         cv2.circle(img, center, radius, (0, 255, 0), 2)

        #     cv2.imshow("Hough Circle Transform", img)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()
            
        
        
        # 2_3mm_no_boxes -> threshold at 175, size (0.001, 0.1), circularity (0.1, 1), compactness (0.1, 1)
        # 2_20mm_no_boxes -> threshoold at 100, size (0.03, 1), circularity (0.03, 1), compactness (0.03, 1)