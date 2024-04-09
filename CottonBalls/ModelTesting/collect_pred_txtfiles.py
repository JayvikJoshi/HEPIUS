from ultralytics import YOLO
import os
import cv2

model = YOLO('/Users/jayvik/Documents/GitHub/HEPIUS/CottonBalls/ModelTesting/cleaned_best.pt')

dir = "/Users/jayvik/Desktop/Data/test_3/partitioned_data/filter_two_images/"
output_dir = "/Users/jayvik/Desktop/Data/test_3/partitioned_data/filter_two_pred/"  # Specify the output folder where you want to save the .txt files

image_paths = []

for file in os.listdir(dir):
    if file.endswith(".png"):
        image_paths.append(os.path.join(dir, file))

for image_path in image_paths:
    img = cv2.imread(image_path)
    results = model(image_path)
    
    txt_filename = os.path.splitext(os.path.basename(image_path))[0] + ".txt"
    txt_filepath = os.path.join(output_dir, txt_filename)
    
    with open(txt_filepath, "w") as txt_file:
        for r in results:
            for box in r.boxes.xyxy:
                x1, y1, x2, y2 = box.tolist()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw rectangle
                # Convert coordinates to YOLO format
                img_h, img_w = img.shape[:2]
                x_center = (x1 + x2) / (2 * img_w)
                y_center = (y1 + y2) / (2 * img_h)
                width = (x2 - x1) / img_w
                height = (y2 - y1) / img_h
                # Write coordinates to the .txt file in YOLO format
                txt_file.write(f"{r.names[0]} {x_center} {y_center} {width} {height}\n")
    
    #cv2.imwrite(os.path.join(output_dir, os.path.basename(image_path)), img)  # Save the image with bounding boxes

