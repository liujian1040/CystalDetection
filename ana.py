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
    print(seg.shape)
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

def fit_circle(contours, label):
    (x, y), radius = cv2.minEnclosingCircle(contours)
    center = (int(x), int(y))
    radius = int(radius)
#    cv2.circle(circle_img, center, radius, ELLIPSE_COLOR[label], 5)
    return radius * 2 * SCALE

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

    return major_axis * SCALE, minor_axis * SCALE

        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_file", type=str, default="./val/IMG_194.png", help="Path to Image File")
    parser.add_argument("--result_file", type=str, default=r"./npy/IMG_194.npy", help="Path to Result File")
    parser.add_argument("--out_file", type=str, default="./ellipse.png", help="Path to Output File")
    parser.add_argument("--scale", type=float, default=5 / 9, help="Image Scale")

    opt = parser.parse_args()
    print(opt)

    img_file = opt.img_file
    result_file = opt.result_file
    out_file = opt.out_file
    SCALE = opt.scale

    img = cv2.imread(img_file) # [1700, 2200, 3]
    ellipse_img = deepcopy(img)
    # cv2.namedWindow("result", 0)
    # cv2.resizeWindow('result', 600, 500)
    # cv2.imshow("result", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    result = np.load(result_file, allow_pickle=True)  # [2, 4]
    bboxes = result[0]
    segs = result[1]

    for category_ind, bboxes_category in enumerate(bboxes):
        # For per category
        segs_category = segs[category_ind]
        for bbox, seg in zip(bboxes_category, segs_category):
            try:
                label = CATEGORY_MAP[category_ind]
                score = bbox[-1]
                location = {"x1": bbox[0], "y1": bbox[1], "x2": bbox[2], "y2": bbox[3]}
                area = get_area(seg=deepcopy(seg))
                circumference, contours = get_circumference(img=deepcopy(img), seg=deepcopy(seg))
                if label == "B" or label == "D":
                    #gray = cv2.cvtColor(seg, cv2.COLOR_BGR2GRAY)
                    #_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
                    #contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    #circle_img = deepcopy(img)
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
            except:
                print('wrong')
                continue

            crystal = {"Label": label, "Score": score, "Location": location, "Area": area,
                       "Circumference": circumference, "Major_Axis": major_Axis, "Minor_Axis": minor_Axis,
                       "Aspect_Ratio": aspect_Ratio, "radius": radius}
            print(crystal)
            print(seg.shape)
            INFO.append(crystal)

    # cv2.namedWindow("result", 0)
    # cv2.resizeWindow('result', 600, 500)
    # cv2.imshow("result", ellipse_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite(out_file, cv2.cvtColor(ellipse_img.astype(np.uint8), cv2.COLOR_RGB2BGR),
                [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

