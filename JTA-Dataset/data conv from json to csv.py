# Annotations
# Each annotation file refers to a specific sequence (e.g. seq_42.json is the annotation file of seq_42.mp4). An annotation file consists of a list of lists, that is a matrix having N rows and 10 columns. Each row of the matrix contains the data of a joint; these data are organized as follows:

# Element	Name	Description
# row[0]	frame number	number of the frame to which the joint belongs
# row[1]	person ID	unique identifier of the person to which the joint belongs
# row[2]	joint type	identifier of the type of joint; see 'Joint Types' subsection
# row[3]	x2D	2D x coordinate of the joint in pixel
# row[4]	y2D	2D y coordinate of the joint in pixel
# row[5]	x3D	3D x coordinate of the joint in meters
# row[6]	y3D	3D y coordinate of the joint in meters
# row[7]	z3D	3D z coordinate of the joint in meters
# row[8]	occluded	1 if the joint is occluded; 0 otherwise
# row[9]	self-occluded	1 if the joint is occluded by its owner; 0 otherwise

# Joint Types
# The associations between numerical identifier and type of joint are the following:

#  0: head_top
#  1: head_center
#  2: neck
#  3: right_clavicle
#  4: right_shoulder
#  5: right_elbow
#  6: right_wrist
#  7: left_clavicle
#  8: left_shoulder
#  9: left_elbow
# 10: left_wrist
# 11: spine0
# 12: spine1
# 13: spine2
# 14: spine3
# 15: spine4
# 16: right_hip
# 17: right_knee
# 18: right_ankle
# 19: left_hip
# 20: left_knee
# 21: left_ankle
########## posenet ######################3
# Id	Part
# 0	nose                      nope
# 1	leftEye                   nope
# 2	rightEye                  nope
# 3	leftEar                   nope
# 4	rightEar                  nope
# 5	leftShoulder              8
# 6	rightShoulder             4
# 7	leftElbow                 9
# 8	rightElbow                5
# 9	leftWrist                 10
# 10	rightWrist            6
# 11	leftHip               19
# 12	rightHip              16
# 13	leftKnee              20
# 14	rightKnee             17
# 15	leftAnkle             21
# 16	rightAnkle            18


import json
import numpy as np
import os

# json_file_path = r'annotations\train\seq_42.json'

# json_file_path = os.path.join(os.getcwd(),json_file_path)

# with open(json_file_path, 'r') as json_file:
#     matrix = json.load(json_file)
#     matrix = np.array(matrix)

# # frame_data = matrix[matrix[:, 0] == 118]                 ########### Get data of frame #118
# # person_data = frame_data[frame_data[:, 1] == 3]            ########### Get data of all the joints of person with ID=3 in frame #118

# # print(person_data)
# # [[ 1.180e+02  3.000e+00  0.000e+00  1.082e+03  3.480e+02  4.300e-01
# #   -6.800e-01  4.100e+00  0.000e+00  0.000e+00]
# #  [ 1.180e+02  3.000e+00  1.000e+00  1.090e+03  3.770e+02  4.700e-01
# #   -5.800e-01  4.150e+00  0.000e+00  0.000e+00]
# #  [ 1.180e+02  3.000e+00  2.000e+00  1.100e+03  4.050e+02  5.100e-01
# #   -4.900e-01  4.200e+00  0.000e+00  0.000e+00]
# #  [ 1.180e+02  3.000e+00  3.000e+00  1.089e+03  4.150e+02  4.700e-01
# #   -4.500e-01  4.180e+00  0.000e+00  0.000e+00]
# #  [ 1.180e+02  3.000e+00  4.000e+00  1.045e+03  4.240e+02  3.100e-01
# #   -4.300e-01  4.270e+00  0.000e+00  0.000e+00]
# #  [ 1.180e+02  3.000e+00  5.000e+00  1.010e+03  4.900e+02  1.900e-01
# #   -1.900e-01  4.320e+00  0.000e+00  0.000e+00]
# #  [ 1.180e+02  3.000e+00  6.000e+00  9.960e+02  5.520e+02  1.300e-01
# #    4.000e-02  4.220e+00  0.000e+00  0.000e+00]
# #  [ 1.180e+02  3.000e+00  7.000e+00  1.109e+03  4.140e+02  5.400e-01
# #   -4.500e-01  4.170e+00  0.000e+00  0.000e+00]
# #  [ 1.180e+02  3.000e+00  8.000e+00  1.159e+03  4.180e+02  7.200e-01
# #   -4.400e-01  4.170e+00  0.000e+00  0.000e+00]
# #  [ 1.180e+02  3.000e+00  9.000e+00  1.199e+03  4.800e+02  8.700e-01
# #   -2.200e-01  4.200e+00  0.000e+00  0.000e+00]
# #  [ 1.180e+02  3.000e+00  1.000e+01  1.219e+03  5.490e+02  9.300e-01
# #    3.000e-02  4.170e+00  0.000e+00  0.000e+00]
# #  [ 1.180e+02  3.000e+00  1.100e+01  1.102e+03  4.720e+02  5.200e-01
# #   -2.500e-01  4.260e+00  0.000e+00  0.000e+00]
# #  [ 1.180e+02  3.000e+00  1.200e+01  1.101e+03  5.020e+02  5.200e-01
# #   -1.400e-01  4.280e+00  0.000e+00  0.000e+00]
# #  [ 1.180e+02  3.000e+00  1.300e+01  1.101e+03  5.260e+02  5.200e-01
# #   -5.000e-02  4.280e+00  0.000e+00  0.000e+00]
# #  [ 1.180e+02  3.000e+00  1.400e+01  1.101e+03  5.490e+02  5.200e-01
# #    3.000e-02  4.280e+00  0.000e+00  0.000e+00]
# #  [ 1.180e+02  3.000e+00  1.500e+01  1.101e+03  5.540e+02  5.200e-01
# #    5.000e-02  4.270e+00  0.000e+00  0.000e+00]
# #  [ 1.180e+02  3.000e+00  1.600e+01  1.073e+03  5.720e+02  4.200e-01
# #    1.200e-01  4.300e+00  0.000e+00  0.000e+00]
# #  [ 1.180e+02  3.000e+00  1.700e+01  1.069e+03  6.800e+02  4.100e-01
# #    5.200e-01  4.320e+00  0.000e+00  0.000e+00]
# #  [ 1.180e+02  3.000e+00  1.800e+01  1.078e+03  7.790e+02  4.500e-01
# #    9.200e-01  4.440e+00  0.000e+00  0.000e+00]
# #  [ 1.180e+02  3.000e+00  1.900e+01  1.125e+03  5.760e+02  6.100e-01
# #    1.300e-01  4.250e+00  0.000e+00  0.000e+00]
# #  [ 1.180e+02  3.000e+00  2.000e+01  1.142e+03  6.850e+02  6.500e-01
# #    5.200e-01  4.140e+00  0.000e+00  0.000e+00]
# #  [ 1.180e+02  3.000e+00  2.100e+01  1.149e+03  7.380e+02  7.300e-01
# #    7.600e-01  4.460e+00  0.000e+00  0.000e+00]]


# # 1920×1080
# # print(matrix[ #frame ][ #element ])
# # matrix = matrix.T
# # print(matrix[ #element ][ #frame number ])

# # for i in range(0,580492):
# #     x_data = matrix[i][]
# x2d = []
# y2d = []
# x3d = []
# y3d = []
# z3d = []

# for i in range(1,901):
#     frame_data = matrix[matrix[:, 0] == i]
#     for j in range(len(frame_data[:,1])):
#         person_data = frame_data[frame_data[:, 1] == j]
#         x3d.append(person_data[:,5])
#         y3d.append(person_data[:,6])
#         z3d.append(person_data[:,7])
#         x2d.append(person_data[:,3])
#         y2d.append(person_data[:,4])
# print(x2d)
# print(y2d[1])
import re
from joint import Joint
from pose import Pose
import pandas as pd
import os
import sys
from tqdm import tqdm




# path = r"C:\Users\shrey_imghzs\Desktop\JTA-Dataset\poses\train\seq_0\1.npy"
path = os.getcwd()
csv_path = os.path.join(path,"csv")
json_path = os.path.join(path,"poses")
# csv_path = r"C:\Users\shrey_imghzs\Desktop\JTA-Dataset\csv"


allowed_num = [4,5,6,8,9,10,16,17,18,19,20,21]
x2d = []
y2d = []
x3d = []
y3d = []
z3d = []
d2d = ""
d3d = ""
dic_of_details = {}
counter = 0
# for person in b:
#     counter += 1
#     for p in range(len(person)):
#         if p in allowed_num:
#             det = f"{person[p]}"
#             det = det.split(sep="|")
#             det.pop()
#             det.pop(0)
#             d2d = str(det[0].split(sep=":").pop()).strip("(),").split(sep=",")
#             d3d = str(det[1].split(sep=":").pop()).strip("(),").split(sep=",")
#             x2d.append(float(d2d[0]))
#             y2d.append(float(d2d[1]))
#             x3d.append(float(d3d[0]))
#             y3d.append(float(d3d[1]))
#             z3d.append(float(d3d[2]))
#     dic_of_details = {"x2d":x2d,"y2d":y2d,"x3d":x3d,"y3d":y3d,"z3d":z3d}
#     df = pd.DataFrame(dic_of_details)
#     df.to_csv(os.path.join(csv_path,f"{counter}.csv"))
#     # print(df)
# files 2,30,400 

pbar = tqdm(total=256)
fi_cnt = 0

# [5:17:39]

for d_num in range(0,256):
    # dir_path = r"C:\Users\shrey_imghzs\Desktop\JTA-Dataset\poses\train\seq_" + f"{d_num}"
    dir_path = os.path.join(json_path,f"train\seq_{d_num}")
    # print(dir_path)
    # break
    fi_cnt += 1
    for file in os.listdir(dir_path):
        f_name = os.path.join(dir_path,file)
        # print(f_name)
        b = np.load(f_name,allow_pickle=True,fix_imports=True)
        for person in b:
            counter += 1
            x2d = []
            y2d = []
            x3d = []
            y3d = []
            z3d = []
            d2d = ""
            d3d = ""
            for p in range(len(person)):
                if p in allowed_num:
                    det = f"{person[p]}"
                    det = det.split(sep="|")
                    det.pop()
                    det.pop(0)
                    d2d = str(det[0].split(sep=":").pop()).strip("(),").split(sep=",")
                    d3d = str(det[1].split(sep=":").pop()).strip("(),").split(sep=",")
                    x2d.append(float(d2d[0]))
                    y2d.append(float(d2d[1]))
                    x3d.append(float(d3d[0]))
                    y3d.append(float(d3d[1]))
                    z3d.append(float(d3d[2]))
            dic_of_details = {"x2d":x2d,"y2d":y2d,"x3d":x3d,"y3d":y3d,"z3d":z3d}
            df = pd.DataFrame(dic_of_details)
            df.to_csv(os.path.join(csv_path,f"{counter}.csv"))
    pbar.update()
pbar.close()