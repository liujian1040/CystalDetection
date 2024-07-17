# -*- coding: utf-8 -*-

import numpy as np
import cv2
from copy import deepcopy
import time
import datetime
import torch
import argparse

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
    
    return length * SCALE, cnts[max_ind]


def fit_ellipse(ellipse_img, contours, label):
    ellipse = cv2.fitEllipse(contours)
    major_axis = max(ellipse[1])
    minor_axis = min(ellipse[1])
    # 油滴用圆形拟合的颜色，默认为(0, 0, 255)
    cv2.ellipse(ellipse_img, ellipse, (0, 0, 255), 5)

    return major_axis, minor_axis

def get_color_mask(img, RGB):
    agg_color = deepcopy(img)
    for c in range(3):
        agg_color[:,:,c] = RGB[c]
    return agg_color
    
def get_total_mask(target_list, mask):
    for target in target_list:
        seg = target['seg']
        seg = seg.astype(np.uint8)
        seg = np.expand_dims(seg, axis=-1)
        seg = torch.tensor(seg)
        seg = seg.expand_as(torch.tensor(deepcopy(img)))
        seg = seg.numpy()
        seg[seg == 1] = 255
        mask = cv2.bitwise_or(mask, seg)
    return mask

def draw_crystal(img, crystal_list, opt):
    bg_reserve = opt.bg_reserve
    #edge_smoothness = opt.edge_smoothness
    with_index = opt.with_index

    a_list = list(filter(lambda crystal: crystal['label']=='A', crystal_list)) #find all aggs
    b_list = list(filter(lambda crystal: crystal['label']=='B', crystal_list)) #find all aggs
    c_list = list(filter(lambda crystal: crystal['label']=='C', crystal_list)) #find all aggs
    d_list = list(filter(lambda crystal: crystal['label']=='D', crystal_list)) #find all aggs
    
    agg_mask = deepcopy(img)
    bubble_mask = deepcopy(img)
    crystal_mask = deepcopy(img)
    drop_mask = deepcopy(img)
    agg_mask[:,:,:] = 0
    bubble_mask[:,:,:] = 0
    crystal_mask[:,:,:] = 0
    drop_mask[:,:,:] = 0

    #assign RGB color for mask
    RGB_MEAN = [np.mean(img[:,:,c]) for c in range(3)]
    agg_color = get_color_mask(img, [0, 255, 255])#yellow
    bubble_color = get_color_mask(img, [0, 0, 255])# blue
    crystal_color = get_color_mask(img, [0, 255, 0]) #green
    drop_color = get_color_mask(img, [255, 0, 255]) #purple
    background_color = get_color_mask(img, RGB_MEAN)
    
    #get total mask for a,b,c,d
    agg_mask = get_total_mask(a_list, agg_mask)
    bubble_mask = get_total_mask(b_list, bubble_mask)
    crystal_mask = get_total_mask(c_list, crystal_mask)
    drop_mask = get_total_mask(d_list, drop_mask)
    
    #get background_mask
    foreground_mask = cv2.bitwise_or(agg_mask, bubble_mask)
    foreground_mask = cv2.bitwise_or(foreground_mask, crystal_mask)
    foreground_mask = cv2.bitwise_or(foreground_mask, drop_mask)
    background_mask = cv2.bitwise_not(foreground_mask)

    #assign certain color for different instance
    agg_mask = cv2.bitwise_and(agg_mask, agg_color)
    bubble_mask = cv2.bitwise_and(bubble_mask, bubble_color)
    crystal_mask = cv2.bitwise_and(crystal_mask, crystal_color)
    drop_mask = cv2.bitwise_and(drop_mask, drop_color)
    background = cv2.bitwise_and(background_mask, background_color)
    
    if not bg_reserve:
        img[background_mask == 255]=0
    tmp1 = cv2.addWeighted(img,0.125, agg_mask, 0.125, 0.5)
    tmp2 = cv2.addWeighted(img,0.25, bubble_mask, 0.25, 0.5)
    tmp3 = cv2.addWeighted(img,0.125,crystal_mask,0.125, 0.5)
    tmp4 = cv2.addWeighted(img,0.25,drop_mask,0.25, 0.5)
    
    result = cv2.add(tmp1, tmp2)
    result = cv2.add(result, tmp3)
    result = cv2.add(result, tmp4)
    if not bg_reserve:
        result = cv2.add(result, background)
    
    result[result[:,:,0]<1,0] = 0
    #cv2.namedWindow("result", 0)
    #cv2.resizeWindow('result', 600, 500)
    #cv2.imshow("result", result)
    #cv2.waitKey(0)
        
    # draw contours    
    for crystal in crystal_list:
        seg = crystal['seg']
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
        
        # make the contours more smooth
        #approx_list = []
        #for cnt in cnts:
        #    approx = cv2.approxPolyDP(cnt,edge_smoothness*cv2.arcLength(cnt,True),True)
        #    approx_list.append(approx)
        #cnts=approx_list

        # draw contours with different colors
        if crystal['label'] == 'A':
            cv2.drawContours(result, cnts, -1, (0, 0, 255), 12)
        if crystal['label'] == 'B':
            cv2.drawContours(result, cnts, -1, (255, 255, 0), 6)
        if crystal['label'] == 'C':
            cv2.drawContours(result, cnts, -1, (0, 255, 0), 12)
        if crystal['label'] == 'D':
            cv2.drawContours(result, cnts, -1, (255, 255, 255), 1) # 白色，画mask边界
            #for cnt in cnts:
                #fit_ellipse(result, cnt, 1) # 拟合的边界
        bbox = crystal['bbox']
        
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        #cv2.rectangle(result, (x1, y1), (x2, y2), (0, 0, 255), 3) # 修改box的厚度
        if with_index:
            cv2.putText(result, str(crystal['index']), (x1+20, y1+20), cv2.FONT_HERSHEY_COMPLEX, 1.0, (255, 255, 255), 1)

    return result
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_file", type=str, default="./1106/img/1.png", help="Path to Image File")
    parser.add_argument("--result_file", type=str, default="./1106/npy/1.npy", help="Path to Result File")
    parser.add_argument("--out_file", type=str, default="./ellipse.png", help="Path to Output File")
    parser.add_argument("--scale", type=float, default=5 / 9, help="Image Scale")
    parser.add_argument("--save_file", type=str, default="./vis/2_vis.jpg", help="Path to Output File")
    parser.add_argument("--bg_reserve", type=bool, default=True, help="reserve background or not")
    parser.add_argument("--with_index", type=bool, default=False, help="draw index for each crystal") # 给目标加序号，改成True
    #parser.add_argument("--edge_smoothness", type=float, default=0.01, help="smoothness of crystal edge")

    opt = parser.parse_args()
    print(opt)

    img_file = opt.img_file
    result_file = opt.result_file
    out_file = opt.out_file
    SCALE = opt.scale

    img = cv2.imread(img_file) # [1700, 2200, 3]
    ellipse_img = deepcopy(img)
    vis_img = deepcopy(img)

    result = np.load(result_file, allow_pickle=True)  # [2, 4]
    bboxes = result[0]
    segs = result[1]
    crystal_list = []
    
    index = 1

    for category_ind, bboxes_category in enumerate(bboxes):
        # For per category
        segs_category = segs[category_ind]
        for bbox, seg in zip(bboxes_category, segs_category):
            label = CATEGORY_MAP[category_ind]
            score = bbox[-1]
            crystal = {'index': index, 'bbox': bbox, 'seg':seg, 'label': label}
            #print(bbox)
            crystal_list.append(crystal)
            index += 1
            
            
    result = draw_crystal(img = deepcopy(img), crystal_list = crystal_list, opt = opt)
    cv2.imwrite(opt.save_file, result)
    

