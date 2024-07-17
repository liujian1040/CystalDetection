# -*- coding: utf-8 -*-
import json
import os
import argparse
from shutil import copy

def is_small(shape):
    points = instance["points"]
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    x_min = min(x)
    y_min = min(y)
    x_max = max(x)
    y_max = max(y)
    
    if (x_max-x_min)*5/9 < 50 or (y_max-y_min)*5/9 < 50:
        return True
        
    return False



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_path", type=str, default=r"C:/Users/Administrator/Desktop/CystalDetection-master/train", help="Path to Image Directory")
    parser.add_argument("--dir_path_small", type=str, default=r"C:/Users/Administrator/Desktop/CystalDetection-master/small", help="Path to Image Directory")
    opt = parser.parse_args()
    print(opt)

    dir_path = opt.dir_path
    small_path = opt.dir_path_small

    l = os.listdir(dir_path)
    l = [i for i in l if i.endswith('.json')]

    for i in l:
        small = 0
        print(i)
        with open(dir_path + "\\" + i, 'r', encoding='utf-8') as f:
            temp = json.loads(f.read())
            anno = temp['shapes']
            for instance in anno:
                if is_small(instance):
                    small += 1

        print(small)
        image_name = i[:-5] + '.png'
        if small > 20:
            copy(dir_path + "\\" + i, small_path + "\\" + i)
            copy(dir_path + "\\" + image_name, small_path + "\\" + image_name)
            


