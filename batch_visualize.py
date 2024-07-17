# -*- coding: utf-8 -*-
import os

            
def single_visualize(img_path, npy_path, save_path):
    cmd = 'python visualize.py --img_file '
    cmd += img_path
    cmd += ' --result_file '
    cmd += npy_path
    cmd += ' --save_file '
    cmd += save_path
    #cmd += ' --bg_reserve '
    #cmd += 'True'
    os.system(cmd)
    
#apply detect.py and get npy file on muti images  
def batch_img_detect(img_dir, npy_dir, save_dir):
    for filename in os.listdir(img_dir):
        if filename.endswith('png'):
            img_path = img_dir + "/" + filename
            npy_path = npy_dir + "/" + filename[:-4]+'.npy'
            save_path = save_dir + "/" + filename[:-4]+'_vis.jpg'
            print("Input image: %s, npy file: %s, save to: %s"%(img_path,npy_path,save_path))
            single_visualize(img_path, npy_path,save_path)
        

if __name__ == '__main__':
    img_dir = './val'
    npy_dir = './npy'
    #npy_dir = 'F:/mask_rcnn_final/npy'
    save_dir = './vis'

    batch_img_detect(img_dir,npy_dir,save_dir)