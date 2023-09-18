# def filter_weighted(contours_dictionary_list, n=-1, area_weight=0, circularity_weight=0, compactness_weight=0):
#     filters = [
#         ("area", area_weight),
#         ("circularity", circularity_weight),
#         ("compactness", compactness_weight)
#     ]
    
#     if n == -1:
#         n = len(contours_dictionary_list)
    
#     for contour in contours_dictionary_list:
#         contour["weighted_score"] = 0
#         for key, weight in filters:
#             key_min = min(contours_dictionary_list, key=lambda x: x[key])[key]
#             key_max = max(contours_dictionary_list, key=lambda x: x[key])[key]
#             contour_values = {
#                 key: (contour[key] - key_min) / (key_max - key_min) if key_max != key_min else 0
#             }
#             contour["weighted_score"] += contour_values[key] * weight
    
#     weight_filtered = sorted(contours_dictionary_list, key=lambda contour: contour["weighted_score"], reverse=True)[:n]
#     return weight_filtered

#############################################################################################################################

# #https://www.authentise.com/post/detecting-circular-shapes-using-contours
# import cv2

# raw_image = cv2.imread(r"C:\Users\jayvi\Desktop\HEPIUS\CottonBalls\Data\threshold_Data\2_3mm_no_box.png")
# cv2.imshow('Original Image', raw_image)
# cv2.waitKey(0)

# bilateral_filtered_image = cv2.bilateralFilter(raw_image, 5, 175, 175)
# cv2.imshow('Bilateral', bilateral_filtered_image)
# cv2.waitKey(0)

# edge_detected_image = cv2.Canny(bilateral_filtered_image, 200, 250)
# cv2.imshow('Edge', edge_detected_image)
# cv2.waitKey(0)

# contours, hierarchy = cv2.findContours(edge_detected_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# contour_list = []
# for contour in contours:
#     approx = cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),True)
#     area = cv2.contourArea(contour)
#     if ((len(approx) > 10) & (len(approx) < 1000) & (area > 30) ):
#         contour_list.append(contour)

# cv2.drawContours(raw_image, contour_list,  -1, (255,0,0), 2)
# cv2.imshow('Objects Detected',raw_image)
# cv2.waitKey(0)
