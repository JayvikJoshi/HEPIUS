import cv2
import numpy as np
import config
#from skimage.feature import greycomatrix, greycoprops

def load_image(image_path):
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

def apply_gaussian_blur(image, kernel_size=(5, 5)):
    return cv2.GaussianBlur(image, kernel_size, 0)

def perform_thresholding(image, threshold_value, max_value=255, threshold_type=cv2.THRESH_BINARY):
    _, binary_threshold = cv2.threshold(image, threshold_value, max_value, threshold_type)
    return binary_threshold

def find_contours(binary_image):
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def size_filter_contours(image, contours, min_area_percent, max_area_percent):
    min_area_threshold = min_area_percent * image.shape[0] * image.shape[1]
    max_area_threshold = max_area_percent * image.shape[0] * image.shape[1]
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area_threshold <= area <= max_area_threshold:
            filtered_contours.append(contour)
    return filtered_contours


def visualize_contours(image, contours):
    image_with_contours = np.copy(image)
    cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 2)
    return image_with_contours

def display_images_opencv(images):
    for i, image in enumerate(images):
        cv2.imshow(f"Image {i+1}" + str(image), image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def image_dimensions(image):
    dimensions = image.shape
    height, width = dimensions
    print(height, width)
    return height, width

def filter_contours_by_circularity(contours, min_threshold, max_threshold):
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter != 0:
            circularity_index = (4 * np.pi * area) / (perimeter * perimeter)
            if min_threshold <= circularity_index <= max_threshold:
                filtered_contours.append(contour)

    return filtered_contours

def filter_contours_by_compactness(contours, min_compactness, max_compactness):
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter != 0:
            compactness_index = area / (perimeter * perimeter)
            if min_compactness <= compactness_index <= max_compactness:
                filtered_contours.append(contour)

    return filtered_contours

def hough_circle_transform(image, dp, min_dist, param1, param2, min_radius, max_radius):
    blurred_image = cv2.GaussianBlur(image, (9, 9), 2)
    circles = cv2.HoughCircles(blurred_image, cv2.HOUGH_GRADIENT, dp, min_dist, param1=param1, param2=param2, minRadius=min_radius, maxRadius=max_radius)
    return circles



if __name__ == "__main__":
    img = load_image(config.input_path)
    
    
    output_list = [img]
    gauss = apply_gaussian_blur(img)
    output_list.append(gauss)
    bin = perform_thresholding(gauss, 100)
    output_list.append(bin)
    contours = find_contours(bin)
    contour_image = visualize_contours(img, contours)
    output_list.append(contour_image)
    contours_filtered_size = size_filter_contours(img, contours, 0.03, 1)
    contour_image_1 = visualize_contours(img, contours_filtered_size)
    output_list.append(contour_image_1) 
    contours_filtered_circularity = filter_contours_by_circularity(contours_filtered_size, 0.3, 1)
    contour_image_2 = visualize_contours(img, contours_filtered_circularity)
    output_list.append(contour_image_2)
    contours_filtered_compactness = filter_contours_by_circularity(contours_filtered_circularity, 0.3, 1)
    contour_image_3 = visualize_contours(img, contours_filtered_compactness)
    output_list.append(contour_image_3)
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
        
        
        
    #2_3mm_no_boxes -> threshold at 175, size (0.001, 0.1), circularity (0.1, 1), compactness (0.1, 1)
    #2_20mm_no_boxes -> threshoold at 100, size (0.03, 1), circularity (0.03, 1), compactness (0.03, 1)