# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
from copy import deepcopy
import time
import datetime
import torch
import argparse
import pandas as pd

# The Length(μm) of One Pixel
SCALE = 5 / 9
# The Area of One Pixel
PIXEL_AREA = SCALE * SCALE
# The All Information of Crystal, e.g. [{}, {}, {}, ..., {}], Details of One Single Crystal as Follows
# {"Label": xxx, "Score": xxx, "Location": xxx, "Area": xxx, "Circumference": xxx, "Major_Axis": xxx, "Minor_Axis": xxx, "Aspect_Ratio": xxx}
INFO = []
# Category Map
CATEGORY_MAP = ["A", "B", "C", "D"]
# RGB of Ellipse for Every Category
ELLIPSE_COLOR = {"A": (0, 0, 255), "B": (255, 0, 0), "C": (0, 255, 0), "D": (255, 0, 255)}



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
    # cv2.namedWindow("result", 0)
    # cv2.resizeWindow('result', 600, 500)
    # cv2.imshow("result", seg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return length * SCALE, cnts[max_ind]


def fit_ellipse(ellipse_img, contours, label):
    ellipse = cv2.fitEllipse(contours)
    major_axis = max(ellipse[1])
    minor_axis = min(ellipse[1])
    cv2.ellipse(ellipse_img, ellipse, ELLIPSE_COLOR[label], 5)
    # cv2.namedWindow("result", 0)
    # cv2.resizeWindow('result', 600, 500)
    # cv2.imshow("result", ellipse_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return major_axis, minor_axis
    
def in_box(agg, crystal):
    crystal_center_x = (crystal['x1']+crystal['x2'])/2
    crystal_center_y = (crystal['y1']+crystal['y2'])/2
    #print(crystal_center_x)
    if crystal_center_x >= agg['x1'] and \
        crystal_center_x <= agg['x2'] and \
        crystal_center_y >= agg['y1'] and \
        crystal_center_y <= agg['y2']:
        return True
        
    return False
    
def single_agg_detect(seg_result_path, img_file):
    crystal_list = []
    agg_list = []
    img = cv2.imread(img_file) # [1700, 2200, 3]
    ellipse_img = deepcopy(img)
    result = np.load(seg_result_path, allow_pickle=True)  # [2, 4]
    bboxes = result[0]
    segs = result[1]

    for category_ind, bboxes_category in enumerate(bboxes):
        # For per category
        segs_category = segs[category_ind]
        for bbox, seg in zip(bboxes_category, segs_category):
            label = CATEGORY_MAP[category_ind]
            score = bbox[-1]
            location = {"x1": bbox[0], "y1": bbox[1], "x2": bbox[2], "y2": bbox[3]}
            area = get_area(seg=deepcopy(seg))
            try:
                circumference, contours = get_circumference(img=deepcopy(img), seg=deepcopy(seg))
            except:
                print("Warning: Fail to get circumference in {}, skip it.".format(img_file))
                continue
            try:
                major_Axis, minor_Axis = fit_ellipse(ellipse_img=ellipse_img, contours=contours, label=label)
            except:
                print("Warning: Fail to get circumference in {}, skip it.".format(img_file))
                continue
            aspect_Ratio = major_Axis / minor_Axis

            crystal = {"file": img_file, "Label": label, "Score": score, 
                       "x1": location['x1'], "x2": location['x2'], "y1": location['y1'], "y2": location['y2'], 
                       "Area": area, "Circumference": circumference, 
                       "Major_Axis": major_Axis, "Minor_Axis": minor_Axis, "Aspect_Ratio": aspect_Ratio}
            if crystal['Label']=='A':
                agg_list.append(crystal)
                print(crystal)
            if crystal['Label']=='C':
                crystal_list.append(crystal)
                print(crystal)
    '''            
    for agg_idx, agg in enumerate(agg_list):
        agg_count = 0
        for crystal in crystal_list:
            if in_box(agg, crystal):
                agg_count = agg_count + 1
        agg_list[agg_idx]['A_degree'] = agg_count
    '''
    for crystal_index, crystal in enumerate(crystal_list):
        crystal_list[crystal_index]['agg'] = 0
    for crystal_index, crystal in enumerate(crystal_list):
        for agg_idx, agg in enumerate(agg_list):
            if in_box(agg, crystal):
                crystal_list[crystal_index]['agg'] = 1
                
    
    return crystal_list

def batch_img_ana(origin_img_dir, seg_result_dir, output_file):
    df = pd.DataFrame()
    for filename in os.listdir(origin_img_dir):
        if filename.endswith('png'):
            origin_img_path = origin_img_dir + "/" + filename
            seg_result_path = seg_result_dir + "/" + filename[:-4]+'.npy'
            print("Input image: %s, npy file: %s"%(origin_img_path,seg_result_path))
            agg_info_list = single_agg_detect(seg_result_path, origin_img_path)
            if len(agg_info_list)>0:
                for agg_info in agg_info_list:
                    temp = pd.DataFrame.from_dict(agg_info, orient='index').T
                    df = pd.concat([df, temp], ignore_index=True)
    
    df.to_excel(output_file)

 

if __name__ == '__main__':
    origin_img_dir = './val'
    seg_result_dir = './npy'
    output_file = 'crystal.xls'

    batch_img_ana(origin_img_dir, seg_result_dir, output_file)