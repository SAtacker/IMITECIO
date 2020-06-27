# PoseNet Python
# Copyright 2018 Ross Wightman

# Posenet tfjs converter (code in posenet/converter)
# Copyright (c) 2017 Infocom TPO (https://lab.infocom.co.jp/)
# Modified (c) 2018 Ross Wightman

# tfjs PoseNet weights and original JS code
# Copyright 2018 Google LLC. All Rights Reserved.


import tensorflow as tf
import cv2
import time
import argparse
import numpy as np
import posenet
import win32api, win32con
import keyboard
import subprocess
import threading
# from pynput.mouse import Button,Controller
# import socket #for client-server communication

# import keyboard

###################################### keypoint co-ords line 52 ##############################
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default=1290)
parser.add_argument('--cam_height', type=int, default=720)
parser.add_argument('--scale_factor', type=float, default=0.7125)
parser.add_argument('--file', type=str, default=None, help="Optionally use a video file instead of a live camera")
args = parser.parse_args()


########################################  angle    ###################################################
def angle(keypoint):
    left_shoulder_xy = np.array([keypoint[5,1],keypoint[5,0]])
    right_shoulder_xy = np.array([keypoint[6,1],keypoint[6,0]])
    left_elbow_xy=np.array([keypoint[7,1],keypoint[7,0]])
    right_elbow_xy=np.array([keypoint[8,1],keypoint[8,0]])
    left_wrist_xy=np.array([keypoint[9,1],keypoint[9,0]])
    right_wrist_xy=np.array([keypoint[10,1],keypoint[10,0]])
    left_hip_xy=np.array([keypoint[11,1],keypoint[11,0]])
    right_hip_xy=np.array([keypoint[12,1],keypoint[12,0]])

    print("shoulder angle left ",angle2(left_hip_xy,left_shoulder_xy,left_elbow_xy))
    print("shoulder angle right ",angle2(right_hip_xy,right_shoulder_xy,right_elbow_xy))
    print("elbow angle left ",angle2(left_shoulder_xy,left_elbow_xy,left_wrist_xy))
    print("elbow angle right",angle2(right_shoulder_xy,right_elbow_xy,right_wrist_xy))
   
    
def angle2(a,b,c):
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    angle=np.degrees(angle)
    print(angle)
    return angle
        


def main():
    
    with tf.Session() as sess:
        model_cfg, model_outputs = posenet.load_model(args.model, sess)
        output_stride = model_cfg['output_stride']

        if args.file is not None:
            cap = cv2.VideoCapture(args.file)
        else:
            cap = cv2.VideoCapture(args.cam_id)
        cap.set(3, args.cam_width)
        cap.set(4, args.cam_height)
        start = time.time()
        
        

        frame_count = 0
     
        while True:
            input_image, display_image, output_scale = posenet.read_cap(
                cap, scale_factor=args.scale_factor, output_stride=output_stride)
            
            

            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
                model_outputs,
                feed_dict={'image:0': input_image}
            )

            pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multi.decode_multiple_poses(
                heatmaps_result.squeeze(axis=0),
                offsets_result.squeeze(axis=0),
                displacement_fwd_result.squeeze(axis=0),
                displacement_bwd_result.squeeze(axis=0),
                output_stride=output_stride,
                max_pose_detections=1,
                min_pose_score=0.15)

            keypoint_coords *= output_scale
            # print(keypoint_scores)
#################### Get keypoint Co-ordinates #######################################################
            # TODO this isn't particularly fast, use GL for drawing and display someday...
            display_image = posenet.draw_skel_and_kp(
                display_image, pose_scores, keypoint_scores, keypoint_coords,
                min_pose_score=0.15, min_part_score=0.1)
################## Pose Classificaton ##################################################################
            # if(keypoint_scores[0][5]>0.1 and keypoint_scores[0][6]>0.1):
            keypoint_coords =  keypoint_coords.squeeze()
            angle(keypoint_coords)
            


            display_image=cv2.imshow('posenet', display_image)
            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                break
            print(f"Average FPS: {frame_count / (time.time() - start) } ",end="\r",flush=True)

if __name__ == "__main__":
    main()