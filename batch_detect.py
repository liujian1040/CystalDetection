import os
import time


# apply detect.py and get npy file on single image
def single_img_detect(origin_img_path, seg_result_path):
    cmd = 'python detect.py --img_file '
    cmd += origin_img_path
    cmd += ' --npy_file '
    cmd += seg_result_path
    os.system(cmd)


# apply detect.py and get npy file on muti images
def batch_img_detect(input_dir, output_dir):
    for filename in os.listdir(input_dir):
        if filename.endswith('png'):
            origin_img_path = input_dir + "/" + filename
            seg_result_path = output_dir + "/" + filename[:-4] + '.npy'
            print("Input image: %s, output npy file: %s" % (origin_img_path, seg_result_path))
            single_img_detect(origin_img_path, seg_result_path)


input_dir = './val'
output_dir = './npy'
# output_dir = 'F:/mask_rcnn_final/npy'
start_time = time.time()
batch_img_detect(input_dir, output_dir)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Detection process took {elapsed_time:.2f} seconds")
