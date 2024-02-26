import heapq
import cv2

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


def draw_bounding_boxes(image, target_boxes, pred_boxes):
    for box in target_boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for box in pred_boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def compute_iou(pred_txt, target_txt):
    
    with open(target_txt, 'r') as file:
        target_boxes_list = []
        for line in file:
            line_data = line.strip().split()
            box = (float(line_data[1]), float(line_data[2]), float(line_data[3]), float(line_data[4]))
            target_boxes_list.append(box)
        
    with open(pred_txt, 'r') as file:
        num_boxes = 0
        pred_boxes_list = []
        for line in file:
            line_data = line.strip().split()
            box = (float(line_data[1]), float(line_data[2]), float(line_data[3]), float(line_data[4]))
            pred_boxes_list.append(box)
            num_boxes += 1

    iou_list = []

    for box1 in target_boxes_list:
        for box2 in pred_boxes_list:
            x1_box1, y1_box1, w_box1, h_box1 = box1
            x1_box2, y1_box2, w_box2, h_box2 = box2

            x_left = max(x1_box1 - w_box1 / 2, x1_box2 - w_box2 / 2)
            y_top = max(y1_box1 - h_box1 / 2, y1_box2 - h_box2 / 2)
            x_right = min(x1_box1 + w_box1 / 2, x1_box2 + w_box2 / 2)
            y_bottom = min(y1_box1 + h_box1 / 2, y1_box2 + h_box2 / 2)

            if x_right < x_left or y_bottom < y_top:
                iou_list.append(0)
            else:
                intersection_area = (x_right - x_left) * (y_bottom - y_top)
                box1_area = w_box1 * h_box1
                box2_area = w_box2 * h_box2
                iou = intersection_area / (box1_area + box2_area - intersection_area)
                iou_list.append(iou)
    print(iou_list)
    print(heapq.nlargest(num_boxes, iou_list))

if __name__ == "__main__":
    img_path = "/Users/jayvik/Desktop/img.png"
    pred_txt = "/Users/jayvik/Desktop/pred.txt"
    target_txt = "/Users/jayvik/Desktop/targ.txt"
    image, target_boxes, pred_boxes = get_info(img_path, target_txt, pred_txt)
    draw_bounding_boxes(image, target_boxes, pred_boxes)