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
import keyboard
###################################### keypoint co-ords line 52 ##############################
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default=1290)
parser.add_argument('--cam_height', type=int, default=720)
parser.add_argument('--scale_factor', type=float, default=0.7125)
parser.add_argument('--file', type=str, default=None, help="Optionally use a video file instead of a live camera")
args = parser.parse_args()

################################### Difining grid pattern and location of center of left and right shoulder 33333333333333333333333
gidd = np.array((args.cam_width//3,args.cam_width//1.5,args.cam_height//3,args.cam_height//1.5))
def grid(center,overlay_image,row,column):
    # print(gidd)
    row_copy=row
    column_copy=column
    if(center[0]>gidd[0] and center[0]<gidd[1]):
        column='center'
    elif(center[0]<gidd[0]):
        column='left'
    elif(center[0]>gidd[1]):
        column='right'
    if(center[1]>gidd[2] and center[1]<gidd[3]):
        row='center'
    elif(center[1]<gidd[2]):
        row='up'
    elif(center[1]>gidd[3]):
        row='bottom'
    # print("row:",row)
    # print("column:",column)
    cv2.putText(overlay_image, row, (50,430), cv2.FONT_HERSHEY_SIMPLEX,3, (0,255,255), 5, cv2.LINE_AA) 
    cv2.putText(overlay_image, column, (50,490), cv2.FONT_HERSHEY_SIMPLEX,3, (0,255,255), 5, cv2.LINE_AA)
    if(row_copy!=row):
        print('row change')
        # if(row_copy=='up'):
        #     if(row=='center'):
        #         keyboard.press_and_release('s')
        #         print(row)
        #     if(row=='bottom'):
        #         keyboard.press_and_release('s')
        #         print(row)
        if(row_copy=='center'):
            if(row=='up'):
                keyboard.press_and_release('w')
                print(row)
            elif(row=='bottom'):
                keyboard.press_and_release('s')
                print(row)
        # elif(row_copy=='bottom'):
        #     if(row=='center'):
        #         keyboard.press_and_release('w')
        #         print(row)
        #     elif(row=='up'):
        #         keyboard.press_and_release('w')
        #         print(row)
    if(column_copy!=column):
        print('column change')
        if(column_copy=='right'):
            if(column=='center'):
                keyboard.press_and_release('a')
                print(column)
            if(column=='left'):
                keyboard.press_and_release('a')
                keyboard.press_and_release('a')
                print(column)
        if(column_copy=='center'):
            if(column=='left'):
                keyboard.press_and_release('a')
                print(column)
            elif(column=='right'):
                keyboard.press_and_release('d')
                print(column)
        elif(column_copy=='left'):
            if(column=='center'):
                keyboard.press_and_release('d')
                print(column)
            elif(column=='right'):
                keyboard.press_and_release('d')
                keyboard.press_and_release('d')
                print(column)
    return row,column


row='center'
column='center'
def main():
    global row
    global column
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
            overlay_image = posenet.draw_skel_and_kp(
                display_image, pose_scores, keypoint_scores, keypoint_coords,
                min_pose_score=0.15, min_part_score=0.1)
################## Pose Classificaton ##################################################################
            if(keypoint_scores[0][5]>0.1 and keypoint_scores[0][6]>0.1 and pose_scores[0]>0.15):
                keypoint_coords =  keypoint_coords.squeeze()
                # keypoint_coords = np.array(keypoint_coords,dtype=np.int64)
                left_shoulder_y = keypoint_coords[5,0]
                left_shoulder_x = keypoint_coords[5,1]
                right_shoulder_y = keypoint_coords[6,0]
                right_shoulder_x = keypoint_coords[6,1]
                centre_x = int(right_shoulder_x+left_shoulder_x)//2
                centre_y = int(right_shoulder_y+left_shoulder_y)//2
                centre = (centre_x,centre_y)
                cv2.circle(overlay_image,centre,10,(255,255,255),-1)
                row,column=grid(centre,overlay_image,row,column)
                
            cv2.line(overlay_image,(430,0),(430,720),(255,255,255),5)
            cv2.line(overlay_image,(860,0),(860,720),(255,255,255),5)
            cv2.line(overlay_image,(0,240),(1290,240),(255,255,255),5)
            cv2.line(overlay_image,(0,480),(1290,480),(255,255,255),5)
            
            overlay_image=cv2.imshow('posenet', overlay_image)
            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                break
            print(f"Average FPS: {frame_count / (time.time() - start) } ",end="\r",flush=True)

if __name__ == "__main__":
    main()