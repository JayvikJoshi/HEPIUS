import heapq
import cv2
import os


def get_info(img_path, target_path, pred_path):
    image = cv2.imread(img_path)

    with open(target_path, 'r') as file:
        target_boxes = []
        for line in file:
            line_data = line.strip().split()
            x, y, w, h = float(line_data[1]), float(line_data[2]), float(line_data[3]), float(line_data[4])
            image_height, image_width = image.shape[:2]
            x1, y1 = int((x - w / 2) * image_width), int((y - h / 2) * image_height)
            x2, y2 = int((x + w / 2) * image_width), int((y + h / 2) * image_height)
            target_boxes.append((x1, y1, x2, y2))
            
    with open(pred_path, 'r') as file:
        pred_boxes = []
        for line in file:
            line_data = line.strip().split()
            x, y, w, h = float(line_data[1]), float(line_data[2]), float(line_data[3]), float(line_data[4])
            image_height, image_width = image.shape[:2]
            x1, y1 = int((x - w / 2) * image_width), int((y - h / 2) * image_height)
            x2, y2 = int((x + w / 2) * image_width), int((y + h / 2) * image_height)
            pred_boxes.append((x1, y1, x2, y2))

    return image, target_boxes, pred_boxes


def draw_bounding_boxes(image, target_boxes, pred_boxes, iou):
    font = cv2.FONT_HERSHEY_SIMPLEX
    for box in target_boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for box in pred_boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(image, f"IoU: {iou}", (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return image

def compute_iou(target_boxes, pred_boxes):

    iou = []
    for box1 in target_boxes:
        for box2 in pred_boxes:
            
            x1_box1, y1_box1, x2_box1, y2_box1 = box1
            x1_box2, y1_box2, x2_box2, y2_box2 = box2

            if x1_box1 > x2_box2 or x2_box1 < x1_box2 or y1_box1 > y2_box2 or y2_box1 < y1_box2:
                iou.append(0)
            else:
                x_left = max(x1_box1, x1_box2)
                x_right = min(x2_box1, x2_box2)
                y_top = max(y1_box1, y1_box2)
                y_bottom = min(y2_box1, y2_box2)

                intersection_area = (x_right - x_left) * (y_bottom - y_top)
                box1_area = (x2_box1 - x1_box1) * (y2_box1 - y1_box1)
                box2_area = (x2_box2 - x1_box2) * (y2_box2 - y1_box2)
                iou_val = intersection_area / (box1_area + box2_area - intersection_area)
                iou.append(round(iou_val, 3))
    return heapq.nlargest(len(target_boxes), iou)

def display(img_folder, target_folder, pred_folder):

    for img_file in os.listdir(img_folder):
        if img_file.endswith(".png"):
            img_path = os.path.join(img_folder, img_file)
            target_path = os.path.join(target_folder, "target_" + img_file)
            pred_path = os.path.join(pred_folder, "predicted_" + img_file)
            
            image, target_boxes, pred_boxes = get_info(img_path, target_path, pred_path)
            iou = compute_iou(target_boxes, pred_boxes)
            while True:
                bb_image = draw_bounding_boxes(image.copy(), target_boxes, pred_boxes, iou)
                cv2.imshow("Bounding Box Image", bb_image)
                #either quit or save image
                key = cv2.waitKey(0)
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    cv2.imwrite(os.path.join(img_folder, "bb_" + img_file), bb_image)
                    break
            cv2.destroyAllWindows()


if __name__ == "__main__":
    # img_path = "/Users/jayvik/Desktop/img.png"
    # pred_txt = "/Users/jayvik/Desktop/pred.txt"
    # target_txt = "/Users/jayvik/Desktop/targ.txt"
    # image, target_boxes, pred_boxes = get_info(img_path, target_txt, pred_txt)
    # bb_image = draw_bounding_boxes(image, target_boxes, pred_boxes)
    # iou = compute_iou(target_boxes, pred_boxes)

    
