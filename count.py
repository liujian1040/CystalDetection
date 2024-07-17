# -*- coding: utf-8 -*-
import json
import os
import argparse



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_path", type=str, default="D:\Cry\\first_datset", help="Path to Image Directory")

    opt = parser.parse_args()
    print(opt)

    dir_path = opt.dir_path

    l = os.listdir(dir_path)
    l = [i for i in l if i.endswith('.json')]
    A_Count = 0
    B_Count = 0
    C_Count = 0
    D_Count = 0
    for i in l:
        with open(dir_path + "\\" + i, 'r', encoding='utf-8') as f:
            temp = json.loads(f.read())
            anno = temp['shapes']
            for instance in anno:
                if instance['label'] == 'A':
                    A_Count += 1
                elif instance['label'] == 'B':
                    B_Count += 1
                elif instance['label'] == 'C':
                    C_Count += 1
                elif instance['label'] == 'D':
                    D_Count += 1

    print(A_Count)
    print(B_Count)
    print(C_Count)
    print(D_Count)


