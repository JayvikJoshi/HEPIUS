import cv2
import numpy as np

def find_bbox_from_given(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_ANYCOLOR)
    lower_yellow = np.array([0, 100, 100])
    upper_yellow = np.array([0, 255, 255])
    yellow_mask = cv2.inRange(image, lower_yellow, upper_yellow)
    contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes_image, bboxes_value = create_bounding_boxes(contours, image)
    print(bboxes_value)
    cv2.imshow('Yellow Bounding Boxes', bboxes_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def create_bounding_boxes(contours, original_image):
    image_with_bboxes = np.copy(original_image)
    bounding_boxes = []

    for contour in contours:
        bounding_rect = cv2.boundingRect(contour)
        bounding_boxes.append(bounding_rect)

        cv2.rectangle(image_with_bboxes, (bounding_rect[0], bounding_rect[1]),
                      (bounding_rect[0] + bounding_rect[2], bounding_rect[1] + bounding_rect[3]),
                      (0, 255, 0), 3)

    return image_with_bboxes, bounding_boxes

find_bbox_from_given("/Users/jayvik/Documents/GitHub/HEPIUS/CottonBalls/datasets/2_3mm_boxes.png")