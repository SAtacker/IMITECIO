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

################################### Difining grid pattern and location of center of left and right shoulder 33333333333333333333333
gidd = np.array((args.cam_width//3,args.cam_width//1.5,args.cam_height//3,args.cam_height//1.5))
horrizontal = args.cam_height//2
def grid(center,display_image,row,column,fwd_bwd,right_hand,left_hand):
    right_hand =np.array(right_hand,dtype=np.int)
    left_hand = np.array(left_hand,dtype=np.int)
    # print(gidd)
    row_copy=row
    column_copy=column
    fwd_bwd_copy=fwd_bwd
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
    
    if left_hand[0]<=horrizontal :
        fwd_bwd = "Backward"
    else : 
        fwd_bwd = "Forward"
    if right_hand[0]<=horrizontal:
        str_no_str = "Strike"
    else : 
        str_no_str = "Non-Strike"
    # print("row:",row)
    # print("column:",column)
    cv2.circle(display_image,(right_hand[1],right_hand[0]),10,(255,255,0),-1)
    cv2.circle(display_image,(left_hand[1],left_hand[0]),10,(255,255,0),-1)
    cv2.circle(display_image,(args.cam_width//2,horrizontal),10,(255,255,255),-1)
    cv2.putText(display_image, fwd_bwd, (600,400), cv2.FONT_HERSHEY_SIMPLEX,3, (0,255,255), 5, cv2.LINE_AA)
    cv2.putText(display_image, str_no_str, (600,490), cv2.FONT_HERSHEY_SIMPLEX,3, (0,255,255), 5, cv2.LINE_AA)
    cv2.putText(display_image, row, (50,430), cv2.FONT_HERSHEY_SIMPLEX,3, (0,255,255), 5, cv2.LINE_AA) 
    cv2.putText(display_image, column, (50,490), cv2.FONT_HERSHEY_SIMPLEX,3, (0,255,255), 5, cv2.LINE_AA)
    

    if(column_copy!=column):
        print('column change')
        if(column_copy=='right'):
            if(column=='center'):
                keyboard.press_and_release('left')
                print(column)
            if(column=='left'):
                keyboard.press_and_release('left')
                keyboard.press_and_release('left')
                print(column)
        if(column_copy=='center'):
            if(column=='left'):
                keyboard.press_and_release('left')
                print(column)
            elif(column=='right'):
                keyboard.press_and_release('right')
                print(column)
        elif(column_copy=='left'):
            if(column=='center'):
                keyboard.press_and_release('right')
                print(column)
            elif(column=='right'):
                keyboard.press_and_release('right')
                keyboard.press_and_release('right')
                print(column)
    if(fwd_bwd_copy!=fwd_bwd and fwd_bwd=='Backward'):
        print('jump')
        keyboard.press('space')
    
    
    

    
    return row,column,fwd_bwd



################# SERVER CLIENT ###################


#I was thinking of using one function that simply connects to the server, 
#and a different function that uses client_sockets to interact with the server.
#But for now, I've made only one function that sends and receives messages until 
#client hits Ctrl+C. The client can run client.py to connect back.

# def interact_with_server(): #add a parameter here and call this function from somewhere in the code.
#     host = "localhost"
#     port = 8052
#     client_socks = socket.socket(socket.AF_INET, socks.SOCK_STREAM) #create client socket
#     client_socks.connect((host, port)) #connect to host on port

#     message = input(">>> ") #sendWhatEverYouWant

#     while True:
#         try:
#             client_socks.send(message.encode()) # 'message' is what you send
#             server_response = client_socks.recv(1024).decode() #convert received bytes to string
#             print("Server: " + server_response) #print server response to terminal
#             message = input(">>> ") #send again whatever you want
#             continue

#         except KeyboardInterrupt:
#             client_socks.close() #disconnect from server
#             print("Disconnected from " + host) 
#             exit(0)

# # -- change made by wh1t3-h4t

# ###################################################



############opening the game###############
def opengame():
    cmd = "NotEndlessRunner.exe"
    subprocess.call(cmd, shell=True) 
    return None 


row='center'
column='center'
jump='forward'

def main():
    global row
    global column
    global jump
    t1=threading.Thread(target=opengame)
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
        t1.start()
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
            # display_image = posenet.draw_skel_and_kp(
            #     display_image, pose_scores, keypoint_scores, keypoint_coords,
            #     min_pose_score=0.15, min_part_score=0.1)
################## Pose Classificaton ##################################################################
            # if(keypoint_scores[0][5]>0.1 and keypoint_scores[0][6]>0.1):
            keypoint_coords =  keypoint_coords.squeeze()
            # keypoint_coords = np.array(keypoint_coords,dtype=np.int64)
            left_shoulder_y = keypoint_coords[5,0]
            left_shoulder_x = keypoint_coords[5,1]
            right_shoulder_y = keypoint_coords[6,0]
            right_shoulder_x = keypoint_coords[6,1]
            centre_x = int(right_shoulder_x+left_shoulder_x)//2
            centre_y = int(right_shoulder_y+left_shoulder_y)//2
            centre = (centre_x,centre_y)
            cv2.circle(display_image,centre,10,(255,255,255),-1)
            # grid(centre,display_image,row,column,right_hand= keypoint_coords[9],left_hand= keypoint_coords[10])
            row,column,jump=grid(centre,display_image,row,column,jump,right_hand= keypoint_coords[9],left_hand= keypoint_coords[10])
            
            cv2.line(display_image,(430,0),(430,720),(255,255,255),5)
            cv2.line(display_image,(860,0),(860,720),(255,255,255),5)
            cv2.line(display_image,(0,240),(1290,240),(255,255,255),5)
            cv2.line(display_image,(0,480),(1290,480),(255,255,255),5)
            
            display_image=cv2.imshow('posenet', display_image)
            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                t1.join()
                cap.release()
                cv2.destroyAllWindows()
                break
            print(f"Average FPS: {frame_count / (time.time() - start) } ",end="\r",flush=True)

if __name__ == "__main__":
    main()