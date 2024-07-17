# -*- coding: utf-8 -*-

import os
from tqdm import tqdm
import argparse
import numpy as np
import json
import base64
import cv2
from copy import deepcopy
import torch


CATEGORY_MAP = ["A", "B", "C", "D"]


def get_points(img, seg):
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

    cnt = cnts[max_ind]
    cnt = np.squeeze(cnt, axis=1)
    if len(cnt) > 10:
        keep_ind = np.linspace(start=0, stop=len(cnt) - 1, num=10).astype(int)
        cnt = cnt[keep_ind]
    cnt = cnt.tolist()
    # cv2.namedWindow("result", 0)
    # cv2.resizeWindow('result', 600, 500)
    # cv2.imshow("result", seg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return cnt



def add_head(json_file):
    json_file["version"] = "4.5.6"
    json_file["flags"] = {}


def add_core(json_file, result, img):
    shapes_list = []
    bboxes = result[0]
    segs = result[1]

    for category_ind, bboxes_category in enumerate(bboxes):
        # For per category
        segs_category = segs[category_ind]
        for bbox, seg in zip(bboxes_category, segs_category):
            label = CATEGORY_MAP[category_ind]
            points = get_points(img=deepcopy(img), seg=deepcopy(seg))

            crystal = {"label": label, "points": points, "group_id": None, "shape_type": "polygon", "flags": {}}
            print(crystal)
            shapes_list.append(crystal)

    json_file["shapes"] = shapes_list




def add_tail(json_file, file_path):
    json_file["imagePath"] = file_path.split("/")[-1]
    with open(file_path, 'rb') as f:
        image = f.read()
        json_file["imageData"] = str(base64.b64encode(image), encoding='utf-8')
    json_file["imageHeight"] = 576
    json_file["imageWidth"] = 704




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_path", type=str, default=r"./TEST", help="Path to Image Directory")
    parser.add_argument("--score_thre", type=float, default=0.4, help="Score Threshold")
    parser.add_argument("--nms_thre", type=float, default=0.3, help="Iou Thresshold for Non-maximum Suppression")

    opt = parser.parse_args()
    print(opt)

    dir_path = opt.dir_path
    score_thre = opt.score_thre
    nms_thre = opt.nms_thre

    img_list = os.listdir(dir_path)
    img_list = [i for i in img_list if i.endswith(".png") or i.endswith(".PNG") or i.endswith(".jpg") or i.endswith(".JPG")]
    print(img_list)

    for file in tqdm(img_list):
        img_path = dir_path + "/" + file
        img = cv2.imread(img_path)
        print("Processing " + img_path)
        os.system("python detect.py --img_file " + img_path + " --score_thre " + str(score_thre) + " --nms_thre " + str(nms_thre))
        result = np.load(r"./result.npy", allow_pickle=True)  # [2, 4]
        json_file = dict()
        add_head(json_file=json_file)
        add_core(json_file=json_file, result=result, img=img)
        add_tail(json_file=json_file, file_path=img_path)

        json.dump(json_file, open(r"./TEST" + "/" + file.split(".")[0] + ".json", 'w'), indent=2)





