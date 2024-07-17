import os
import numpy as np
import cv2
from copy import deepcopy
import time
import datetime
import torch
import argparse
import pandas as pd
import time


# The Length(μm) of One Pixel
SCALE = 5 / 9
#SCALE =1.79
# The Area of One Pixel
PIXEL_AREA = SCALE * SCALE
# {"Label": xxx, "Score": xxx, "Location": xxx, "Area": xxx, "Circumference": xxx, "Major_Axis": xxx, "Minor_Axis": xxx, "Aspect_Ratio": xxx}
# Category Map
CATEGORY_MAP = ["A", "B", "C", "D"]
# CATEGORY_MAP = ["C"]

def get_area(seg):
    # [1700, 2200]
    pixel_cnt = 0
    for row in seg:
        for pixel in row:
            if pixel:
                pixel_cnt += 1

    return PIXEL_AREA * pixel_cnt
    
def get_circumference(img, seg):
    # [1700, 2200, 3]
    # [1700, 2200]
    seg = seg.astype(np.uint8)
    seg = np.expand_dims(seg, axis=-1)
    img = torch.tensor(img)
    seg = torch.tensor(seg)
    seg = seg.expand_as(img)
    seg = seg.numpy()
    img = img.numpy()
    seg[seg == 1] = 255
    seg = cv2.cvtColor(seg, cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(seg, cv2.COLOR_RGB2GRAY)
    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    cnts, hiera = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # cv2.drawContours(seg, cnts, -1, (0, 0, 255), 3)
    num_points = -1
    max_ind = 0
    for i, cnt in enumerate(cnts):
        if cnt.shape[0] > num_points:
            num_points = cnt.shape[0]
            max_ind = i
    
    length = cv2.arcLength(cnts[max_ind], True)

    return length * SCALE, cnts[max_ind]

def fit_ellipse(ellipse_img, contours, label):
    ellipse = cv2.fitEllipse(contours)
    major_axis = max(ellipse[1])
    minor_axis = min(ellipse[1])
    #cv2.ellipse(ellipse_img, ellipse, ELLIPSE_COLOR[label], 5)
    # cv2.namedWindow("result", 0)
    # cv2.resizeWindow('result', 600, 500)
    # cv2.imshow("result", ellipse_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return major_axis*SCALE, minor_axis*SCALE

def fit_circle(contours, label):
    (x, y), radius = cv2.minEnclosingCircle(contours)
    center = (int(x), int(y))
    radius = int(radius)
    #cv2.circle(circle_img, center, radius, ELLIPSE_COLOR[label], 5)
    return radius * 2 * SCALE

#apply detect.py and get npy file on single image
def single_img_ana(seg_result_path, img_file):
    
    img = cv2.imread(img_file) # [1700, 2200, 3]
    ellipse_img = deepcopy(img)
    result = np.load(seg_result_path, allow_pickle=True)  # [2, 4]
    bboxes = result[0]
    segs = result[1]
    crystal_list = []
    for category_ind, bboxes_category in enumerate(bboxes):
        # For per category
        segs_category = segs[category_ind]
        for bbox, seg in zip(bboxes_category, segs_category):
            label = CATEGORY_MAP[category_ind]
            score = bbox[-1]
            location = {"x1": bbox[0], "y1": bbox[1], "x2": bbox[2], "y2": bbox[3]}
            area = get_area(seg=deepcopy(seg))
            circumference, contours = get_circumference(img=deepcopy(img), seg=deepcopy(seg))
            if label == "B" or label == "D":
                radius = fit_circle(contours, label)
                aspect_Ratio = 0
                major_Axis = 0
                minor_Axis = 0
                if radius is None:
                    print(f"Skipping due to insufficient contour points for label {label}")
                    continue
            else:
                major_Axis, minor_Axis = fit_ellipse(ellipse_img=ellipse_img, contours=contours, label=label)
                aspect_Ratio = major_Axis / minor_Axis
                radius = 0
            try:
                # 如果seg二值化之后全0，则会报错，此时直接跳过即可
                circumference, contours = get_circumference(img=deepcopy(img), seg=deepcopy(seg))

            except:
                print("Warning: Fail to get circumference in {}, skip it.".format(img_file))
                continue
            try:
                major_Axis, minor_Axis = fit_ellipse(ellipse_img=ellipse_img, contours=contours, label=label)
            except:
                print("Warning: Fail to fit ellipse in {}, skip it.".format(img_file))
                continue
            aspect_Ratio = major_Axis / minor_Axis

            crystal = {"file": img_file, "Label": label, "Score": score, 
                       "x1": location['x1'], "x2": location['x2'], "y1": location['y1'], "y2": location['y2'], 
                       "Area": area, "Circumference": circumference, 
                       "Major_Axis": major_Axis, "Minor_Axis": minor_Axis, "Aspect_Ratio": aspect_Ratio, "radius": radius}
            
            crystal_list.append(crystal)
            
    return crystal_list
    

#apply detect.py and get npy file on muti images  
def batch_img_ana(origin_img_dir, seg_result_dir, output_file):
    df = pd.DataFrame()
    for filename in os.listdir(origin_img_dir):
        if filename.endswith('png'):
            origin_img_path = origin_img_dir + "/" + filename
            seg_result_path = seg_result_dir + "/" + filename[:-4]+'.npy'
            print("Input image: %s, npy file: %s"%(origin_img_path,seg_result_path))
            crystal_info_list = single_img_ana(seg_result_path, origin_img_path)
            for crystal_info in crystal_info_list:
                temp = pd.DataFrame.from_dict(crystal_info, orient='index').T
                df = pd.concat([df, temp], ignore_index=True)
    
    df.to_excel(output_file)


origin_img_dir = './val'
seg_result_dir = './npy'
output_file = '1.xls'

start_time = time.time()
batch_img_ana(origin_img_dir, seg_result_dir, output_file)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Detection process took {elapsed_time:.2f} seconds")