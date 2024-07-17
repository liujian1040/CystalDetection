import os
import numpy as np
from mmdet.apis import init_detector, inference_detector
from mmdet.core.post_processing import multiclass_nms
import mmcv
import torch
import argparse
import time
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str,
                        default="./mmdetection/configs/mask_rcnn_r50_caffe_fpn_mstrain-poly_1xSF.py",
                        help="Path to Config File")
    parser.add_argument("--checkpoint_file", type=str,
                        default="./mmdetection/work_dirs/mask_rcnn_r50_caffe_fpn_mstrain-poly_1xSF/latest.pth",
                        help="Path to Checkpoint File")
    parser.add_argument("--img_file", type=str, default="./val2/IMG_876.png", help="Path to Image File")
    parser.add_argument("--npy_file", type=str, default="result", help="Path to npy File")
    parser.add_argument("--out_file", type=str, default="./c5.jpg", help="Path to Output File")
    parser.add_argument("--score_thre", type=float, default=0.5, help="Score Threshold")
    parser.add_argument("--nms_thre", type=float, default=0.7, help="Iou Thresshold for Non-maximum Suppression")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")

    opt = parser.parse_args()
    print(opt)

    config_file = opt.config_file
    checkpoint_file = opt.checkpoint_file
    img = opt.img_file
    out_file = opt.out_file
    score_thre = opt.score_thre
    nms_thre = opt.nms_thre
    gpu_id = opt.gpu_id
    npy_file = opt.npy_file

    #start_time = time.time()
    model = init_detector(config_file, checkpoint_file, device='cuda:' + str(gpu_id))
    result = inference_detector(model, img)
    # 也可以把结束位置放在这里，如果现在的结果不好
    end_time = time.time()

    bbox = result[0]
    seg = result[1]

    multi_scores = []
    multi_bboxes = []

    for c, i in enumerate(bbox):
        # For Per Class
        if len(i) == 0:
            continue
        # [n, 5]
        multi_scores = []
        multi_bboxes = []
        for b in i:
            multi_scores.append([b[-1], 0])
            multi_bboxes.append([b[0], b[1], b[2], b[3]])

        multi_scores = torch.Tensor(multi_scores)
        multi_bboxes = torch.Tensor(multi_bboxes)

        _, _, keep = multiclass_nms(multi_bboxes=multi_bboxes, multi_scores=multi_scores, score_thr=score_thre,
                                     nms_cfg={"iou_threshold": nms_thre, "class_agnostic": False}, return_inds=True)

        multi_scores = multi_scores[keep].numpy()
        multi_bboxes = multi_bboxes[keep].numpy()
        i = i[keep]

        seg_c = np.array(result[1][c])
        seg_c = seg_c[keep]
        if len(keep) == 1:
            seg_c = np.array([seg_c])
            i = np.array([i.tolist()])
        seg_c = [s for s in seg_c]
        result[1][c] = seg_c

        result[0][c] = i

    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # print(f"Detection process took {elapsed_time:.2f} seconds")
    np.save(npy_file, result)
    model.show_result(img, result, out_file=out_file)
