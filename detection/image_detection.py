# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import cv2
import mmcv
import torch
import requests

from mmdet.apis import inference_detector, init_detector
from mmdet.registry import VISUALIZERS

def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection image demo')
    parser.add_argument('image', help='image file')
    # parser.add_argument('config', help='Config file')
    # parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='Bbox score threshold')
    parser.add_argument('--out', action='store_true',help='Output image file')
    parser.add_argument('--show', action='store_true')
    parser.add_argument(
        '--wait-time',
        type=float,
        default=1,
        help='The interval of show (s), 0 is block')
    args = parser.parse_args()
    return args

config = "configs/forestfire_detection/ff_model.py"
checkpoint = "work_dirs/ff_model/checkpoint.pth"

def main():
    args = parse_args()
    assert args.out or args.show, \
        ('Please specify at least one operation (save/show the '
         'image) with the argument "--out" or "--show"')
    
    # build the model from a config file and a checkpoint file
    model = init_detector(config, checkpoint, device=args.device)

    # init visualizer
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    # the dataset_meta is loaded from the checkpoint and
    # then pass to the model in init_detector
    visualizer.dataset_meta = model.dataset_meta

    img = cv2.imread(args.image)
    image_writer = None

    result = inference_detector(model, img)

    img = mmcv.imconvert(img , 'bgr', 'rgb')
    visualizer.add_datasample(
        name='result',
        image=img,
        data_sample=result,
        draw_gt=False,
        pred_score_thr=args.score_thr,
        show=False)
    img = visualizer.get_image()
    img = mmcv.imconvert(img , 'bgr', 'rgb')
    
    if args.show:
        cv2.namedWindow('image', 0)
        cv2.imshow('image',img )
        ch = cv2.waitKey(0)
        if ch == 27 or ch == ord('q') or ch == ord('Q'):
            exit()



    if args.out:
            fire_save = 'fire_save/image/1.jpg'
            cv2.imwrite(fire_save,img)

    if image_writer:
        image_writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
