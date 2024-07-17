from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import skimage.io as io
import pylab,json
import sys
import os

file = open('segm.txt','a+')
sys.stdout = file

#apply detect.py and get npy file on muti images  
def batch_AR(gt, dt_dir):
    path_list = os.listdir(dt_dir)
    path_list.sort(key=lambda x:int(x.split('.')[0])) #对‘.’进行切片，并取列表的第一个值（左边的文件名）转化整数型
    for filename in path_list:
        if filename.endswith('.segm.json'):
            print("eval:"+filename)
            single_AR(gt, dt_dir+"/"+filename)
            
def single_AR(gt,dt):
    cocoGt = COCO(gt)       #标注文件的路径及文件名，json文件形式
    cocoDt = cocoGt.loadRes(dt)  #自己的生成的结果的路径及文件名，json文件形式
    cocoEval = COCOeval(cocoGt, cocoDt, "segm")
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    
def single_test(epoch):
    cmd = "python ./mmdetection/tools/test.py\
        ./mmdetection/configs/mask_rcnn_r50_caffe_fpn_mstrain-poly_1xSF.py\
        ./mmdetection/work_dirs/mask_rcnn_r50_caffe_fpn_mstrain-poly_1xSF/epoch_{}.pth\
        --format-only \
        --eval-options \"jsonfile_prefix=./test_result/{}\" ".format(epoch, epoch)
    os.system(cmd)
        
def batch_test(max_epoch):
    for i in range(1,max_epoch):
        single_test(i)

if __name__ == "__main__":
    batch_test(61)
    batch_AR('val.json', './test_result')
    
