import cv2
import os
import numpy as np
from shutil import copy
import json
from base64 import b64encode

json_list = []
img_list = []

def label_match_img(old_json_path, new_json_path, img_name):
    with open(old_json_path,'r') as load_f:
        load_dict = json.load(load_f)
        load_dict['imagePath'] = img_name
        with open(new_json_path, 'w+') as f:
                json.dump(load_dict, f)
            
def copy_json_label(input_dir_list, output_dir):
    
    for input_dir in input_dir_list:
        count = 0
        for filename in os.listdir(input_dir):
            if filename.endswith('json'):
                origin_file_path = input_dir + "/" + filename
                json_list.append(origin_file_path)
                count += 1
            elif filename.endswith('png'):
                origin_file_path = input_dir + "/" + filename
                img_list.append(origin_file_path)
        print('Scanned: %s, %d samples in this directory.' %(input_dir,count))

    index = 0
    for old_json_path in json_list:
        new_json_path = output_dir + "/" + str(index) + '.json'
        old_img_path = old_json_path[:-4] + 'png'
        new_img_path = output_dir + "/" + str(index) + '.png'
        print(old_json_path)
        print(new_json_path)
        print(old_img_path)
        print(new_img_path)
        #copy(old_json_path, new_json_path)
        copy(old_img_path, new_img_path)
        label_match_img(old_json_path, new_json_path, str(index) + '.png')
        index += 1
    
    print('Merge Complete, %d samples in total.' %(index))
        

input_dir_list = [
    #r'C:/Users/Administrator/Desktop/CystalDetection-master/6',
    r'C:/Users/Administrator/Desktop/CystalDetection-master/2',
    r'C:/Users/Administrator/Desktop/CystalDetection-master/3',
    #r'C:/Users/Administrator/Desktop/CystalDetection-master/4',
    #r'C:/Users/Administrator/Desktop/CystalDetection-master/5',
 ]
output_dir = r'C:\Users\Administrator\Desktop\CystalDetection-master\train'
ext='.png'

copy_json_label(input_dir_list, output_dir)