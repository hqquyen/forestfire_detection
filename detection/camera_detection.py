# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import cv2
import mmcv
import torch
import requests

from mmdet.apis import inference_detector, init_detector
from mmdet.registry import VISUALIZERS


def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection webcam demo')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--score-thr', type=float, default=0.5, help='bbox score threshold')
    args = parser.parse_args()
    return args

#write text to frame
def warning_text(frame,text):
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    # org
    org = (50, 50)
    # fontScale
    fontScale = 1
    # Blue color in BGR
    color = (0, 0, 255)
    # Line thickness of 2 px
    thickness = 2
    frame_text = cv2.putText(frame, text, org, font, 
                fontScale, color, thickness, cv2.LINE_AA)
    return frame_text

#connect to server
def send_server(fire_save, detection_rate):
    head = {'Authorization': 'Token {}'.format("9ea67aae3650b67b0ba3495b28108c113a61e5b1")}

    r = requests.request("post", url="http://skysys.iptime.org:8000/detection/", 
    headers=head, files={"image": open(fire_save, 'rb')},
    data={"detection_rate": detection_rate, "class_name": "Fire", "ai_model": "ubinet"})

    return r



def main():
    config = "configs/forestfire_detection/ff_model.py"
    checkpoint = "work_dirs/ff_model/checkpoint.pth"
    args = parse_args()

    # build the model from a config file and a checkpoint file
    device = torch.device(args.device)
    model = init_detector(config, checkpoint, device=device)

    # init visualizer
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    # the dataset_meta is loaded from the checkpoint and
    # then pass to the model in init_detector
    visualizer.dataset_meta = model.dataset_meta

    camera = cv2.VideoCapture(1)
    if not camera.isOpened():
        print("Cannot open camera")
        exit()
    print('Press "Esc", "q" or "Q" to exit.')
    i = 0
    while True:
        ret_val, img = camera.read()
        result = inference_detector(model, img)
        if len(result.pred_instances.scores) > 0 :
            if result.pred_instances.scores[0] >= args.score_thr:
                detection_rate = "{:.2f}".format(result.pred_instances.scores[0]*100)
                img = mmcv.imconvert(img, 'bgr', 'rgb')
                visualizer.add_datasample(
                name='result',
                image=img,
                data_sample=result,
                draw_gt=False,
                pred_score_thr=args.score_thr,
                show=False)

                img = visualizer.get_image()
                img = mmcv.imconvert(img, 'bgr', 'rgb')
                img = warning_text(img,'FIRE')
                fire_save = 'fire_save/camera/fire_' + str(i) + '.jpg'
                cv2.imwrite(fire_save,img)
                print('\nFIRE') 
                i=i+1
                #send_server(fire_save, detection_rate)

        img = mmcv.imconvert(img, 'bgr', 'rgb')
        visualizer.add_datasample(
            name='result',
            image=img,
            data_sample=result,
            draw_gt=False,
            pred_score_thr=args.score_thr,
            show=False)

        img = visualizer.get_image()
        img = mmcv.imconvert(img, 'bgr', 'rgb')
        cv2.imshow('result', img)

        ch = cv2.waitKey(1)
        if ch == 27 or ch == ord('q') or ch == ord('Q'):
            break


if __name__ == '__main__':
    main()
