import tensorflow as tf
from matplotlib.animation import FuncAnimation
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import cv2
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import math
import threading

cam_width = 1290
cam_height = 720
cam_id = 0
output_stride = 32
############### This will be done at last ######################
parser = argparse.ArgumentParser()
args = parser.parse_args()
############### Part Ids and their names are stated on Official Google TF Posenet Website ###########
# nose = 0
# leftEye = 1
# rightEye = 2
# leftEar = 3
# rightEar = 4
# leftShoulder = 5
# rightShoulder = 6
# leftElbow = 7
# rightElbow = 8
# leftWrist = 9
# rightWrist = 10
# leftHip = 11
# rightHip = 12
# leftKnee = 13
# rightKnee = 14
# leftAnkle = 15
# rightAnkle = 16
###################### Creating a list of part names #################################
partNames = [
    "nose", "leftEye", "rightEye", "leftEar", "rightEar", "leftShoulder",
    "rightShoulder", "leftElbow", "rightElbow", "leftWrist", "rightWrist",
    "leftHip", "rightHip", "leftKnee", "rightKnee", "leftAnkle", "rightAnkle"
]
partNamesZ={
    "nose":0, "leftEye":0, "rightEye":0, "leftEar":0, "rightEar":0, "leftShoulder":0,
    "rightShoulder":0, "leftElbow":0, "rightElbow":0, "leftWrist":0, "rightWrist":0,
    "leftHip":0, "rightHip":0, "leftKnee":0, "rightKnee":0, "leftAnkle":0, "rightAnkle":0
}
##################### Defining a chain / tree based on nose as root point #############
# poseChain = [
#     ["nose", "leftEye"], ["leftEye", "leftEar"], ["nose", "rightEye"],
#     ["rightEye", "rightEar"], ["nose", "leftShoulder"],
#     ["leftShoulder", "leftElbow"], ["leftElbow", "leftWrist"],
#     ["leftShoulder", "leftHip"], ["leftHip", "leftKnee"],
#     ["leftKnee", "leftAnkle"], ["nose", "rightShoulder"],
#     ["rightShoulder", "rightElbow"], ["rightElbow", "rightWrist"],
#     ["rightShoulder", "rightHip"], ["rightHip", "rightKnee"],
#     ["rightKnee", "rightAnkle"]
# ]
################# Defining Connected Parts of skeleton ########################
# connectedPartNames = [
#   ['leftHip', 'leftShoulder'], ['leftElbow', 'leftShoulder'],
#   ['leftElbow', 'leftWrist'], ['leftHip', 'leftKnee'],
#   ['leftKnee', 'leftAnkle'], ['rightHip', 'rightShoulder'],
#   ['rightElbow', 'rightShoulder'], ['rightElbow', 'rightWrist'],
#   ['rightHip', 'rightKnee'], ['rightKnee', 'rightAnkle'],
#   ['leftShoulder', 'rightShoulder'], ['leftHip', 'rightHip']
# ]

######################### Defining Part Ids as a dictionary(parts corresponding to their ids) ###########################3
# partIds = {}
# for i in range(len(partNames)):
#         partIds[partNames[i]] = i
######################## Defining Connected PartIDs corresponding to part names ########################## 

# connectedPartIndices = []

# def part_indices(connected_part_names, dict_part_ids,connected_part_indices):
#     for jointNameA, jointNameB in connected_part_names:
#         connected_part_indices.append([dict_part_ids[jointNameA],dict_part_ids[jointNameB]])

# part_indices(connectedPartNames, partIds, connectedPartIndices)

# print(connectedPartIndices)
# [[11, 5], [7, 5], [7, 9], [11, 13], [13, 15], [12, 6], [8, 6], [8, 10], [12, 14], [14, 16], [5, 6], [11, 12]]

########################### Defiing Parent Child nodes for displacement gradients ####################
# parentChildrenTuples = []

# for joint_name in poseChain:
#     parent_joint_name = joint_name[0]
#     child_joint_name = joint_name[1]
#     parentChildrenTuples.append([partIds[parent_joint_name],partIds[child_joint_name]])

# ######################## Defining the sixteen edges of skeleton ######################
# parentToChildEdges = []
# for joint_id in parentChildrenTuples:
#     parentToChildEdges.append(joint_id[1])

# childToParentEdges = []
# for joint_id in parentChildrenTuples:
#     childToParentEdges.append(joint_id[0])

#################################### Defining connected keypoints names ##############################
# ConnectedKeyPointsNames = {
#     'leftHipleftShoulder':(0,0,255), 'leftShoulderleftHip':(0,0,255),
#     'leftElbowleftShoulder':(255,0,0), 'leftShoulderleftElbow':(255,0,0),
#     'leftElbowleftWrist':(0,255,0), 'leftWristleftElbow':(0,255,0),
#     'leftHipleftKnee':(0,0,255), 'leftKneeleftHip':(0,0,255),
#     'leftKneeleftAnkle':(255,255,0), 'leftAnkleleftKnee':(255,255,0),
#     'rightHiprightShoulder':(0,255,0), 'rightShoulderrightHip':(0,255,0),
#     'rightElbowrightShoulder':(255,0,0), 'rightShoulderrightElbow':(255,0.0),
#     'rightElbowrightWrist':(255,255,0), 'rightWristrightElbow':(255,255,0),
#     'rightHiprightKnee':(255,0,0), 'rightKneerightHip':(255,0,0),
#     'rightKneerightAnkle':(255,0,0), 'rightAnklerightKnee':(255,0,0),
#     'leftShoulderrightShoulder':(0,255,0), 'rightShoulderleftShoulder':(0,255,0),
#     'leftHiprightHip':(0,0,255), 'rightHipleftHip':(0,0,255)
# }


############### ADDING APPROX LENGTHS TO THE BODY's CONNECTED KEYPOINTS ###############################################
ConnectedKeyPointsNames = {
    'leftHipleftShoulder':120, 'leftShoulderleftHip':120,
    'leftElbowleftShoulder':75, 'leftShoulderleftElbow':75,
    'leftElbowleftWrist':70, 'leftWristleftElbow':70,
    'leftHipleftKnee':85, 'leftKneeleftHip':85,
    'leftKneeleftAnkle':40, 'leftAnkleleftKnee':40,
    'rightHiprightShoulder':120, 'rightShoulderrightHip':120,
    'rightElbowrightShoulder':75, 'rightShoulderrightElbow':75,
    'rightElbowrightWrist':70, 'rightWristrightElbow':70,
    'rightHiprightKnee':85, 'rightKneerightHip':85,
    'rightKneerightAnkle':40, 'rightAnklerightKnee':40,
    'leftShoulderrightShoulder':100, 'rightShoulderleftShoulder':100,
    'leftHiprightHip':80, 'rightHipleftHip':80
}

######################### Further Algorithm ###########################################
################### For single pose ###################################
def get_heatmap_scores(heatmaps_result):
    height, width, depth = heatmaps_result.shape
    reshaped_heatmap = np.reshape(heatmaps_result, [height * width, depth])
    coords = np.argmax(reshaped_heatmap, axis=0)
    y_coords = coords // width
    x_coords = coords % width
    return np.concatenate([np.expand_dims(y_coords, 1), np.expand_dims(x_coords, 1)], axis=1)

def get_points_confidence(heatmaps, coords):
    result = []
    for keypoint in range(len(partNames)):
        # Get max value of heatmap for each keypoint
        result.append(heatmaps[coords[keypoint, 0],coords[keypoint, 1], keypoint])
    return result

def get_offset_vectors(coords, offsets_result):
    result = []
    for keypoint in range(len(partNames)):
        heatmap_y = coords[keypoint, 0]
        heatmap_x = coords[keypoint, 1]

        offset_y = offsets_result[heatmap_y, heatmap_x, keypoint]
        offset_x = offsets_result[heatmap_y, heatmap_x, keypoint + len(partNames)]

        result.append([offset_y, offset_x])
    # print(result)
    return result

def get_offset_points(coords, offsets_result, output_stride=output_stride):
    offset_vectors = get_offset_vectors(coords, offsets_result)
    scaled_heatmap = coords * output_stride
    return scaled_heatmap + offset_vectors

def decode_single_pose(heatmaps, offsets, output_stride=output_stride, width_factor=cam_width/257, height_factor=cam_height/257):
    poses = []
    heatmaps_coords = get_heatmap_scores(heatmaps)
    offset_points = get_offset_points(heatmaps_coords, offsets, output_stride)
    keypoint_confidence = get_points_confidence(heatmaps, heatmaps_coords)

    keypoints = [{
        "position": {
            "y": offset_points[keypoint, 0]*height_factor,
            "x": offset_points[keypoint, 1]*width_factor
        },
        "part": partNames[keypoint],
        "score": score
    } for keypoint, score in enumerate(keypoint_confidence)]                        ############### refer https://stackoverflow.com/questions/11479392/what-does-a-for-loop-within-a-list-do-in-python & https://www.geeksforgeeks.org/enumerate-in-python/

    poses.append({"keypoints": keypoints, \
                  "score": (sum(keypoint_confidence) / len(keypoint_confidence))})
    # print(poses)
    return poses

confidence_threshold = 0.1
def drawKeypoints(body, img, color):
    for keypoint in body['keypoints']:
        if keypoint['score'] >= confidence_threshold:
            center = (int(keypoint['position']['x']), int(keypoint['position']['y']))
            radius = 5
            color = color
            cv2.circle(img, center, radius, color, -1, 8)
    
    return None

HeaderPart = {'nose', 'leftEye', 'leftEar', 'rightEye', 'rightEar'}
def drawSkeleton(body, img):
    valid_name = set()
    keypoints = body['keypoints']
    thickness = 2
    for i in range(len(keypoints)):
        src_point = keypoints[i]
        if src_point['part'] in HeaderPart or src_point['score'] < confidence_threshold:
            continue
        for dst_point in keypoints[i:]:
            if dst_point['part'] in HeaderPart or dst_point['score'] < confidence_threshold:
                continue
            name = src_point['part'] + dst_point['part']
            def check_and_drawline(name):
                if name not in valid_name and name in ConnectedKeyPointsNames:
                    color = (255,255,0)#ConnectedKeyPointsNames[name]
                    cv2.line(img, \
                             (int(src_point['position']['x']), int(src_point['position']['y'])), \
                             (int(dst_point['position']['x']), int(dst_point['position']['y'])), \
                             color, thickness)
                    valid_name.add(name)
            name = src_point['part'] + dst_point['part']
            check_and_drawline(name)
            name = dst_point['part'] + src_point['part']
            check_and_drawline(name)
    return None
###############################   MATPLOTLIB SKELETON   ####################################################################
src_z=0
dst_z=0
def drawmatplotlib(body,fig):
    valid_name = set()
    keypoints = body['keypoints']
    print(keypoints)
    plt.cla()
    ax = fig.add_subplot(111,projection='3d')
    for i in range(len(keypoints)):
        src_point = keypoints[i]
        if src_point['part'] in HeaderPart:# or src_point['score'] < confidence_threshold:
            continue
        for dst_point in keypoints[i:]:
            if dst_point['part'] in HeaderPart:# or dst_point['score'] < confidence_threshold:
                continue
            name = src_point['part'] + dst_point['part']
            def check_and_drawline(name):
                global src_z
                global dst_z
                if name not in valid_name and name in ConnectedKeyPointsNames:
                    x=[int(src_point['position']['x']),int(dst_point['position']['x'])]
                    y=[int(src_point['position']['y']),int(dst_point['position']['y'])]
                    print(name,": ")
                    length=math.sqrt((src_point['position']['x']-dst_point['position']['x'])**2+(src_point['position']['y']-dst_point['position']['y'])**2)
                    print(length)
                    dst_z=src_z+math.sqrt(abs(ConnectedKeyPointsNames[name]**2-(src_point['position']['x']-dst_point['position']['x'])**2-(src_point['position']['y']-dst_point['position']['y'])))
                    z=[int(src_z),int(dst_z)]
                    print(z)
                    if(partNamesZ[src_point['part']]==0):
                        partNamesZ[src_point['part']]=src_z
                        src_z=dst_z
                    # print("z:" ,partNamesZ[src_point['part']])
                    # # print(dst_point['part'])
                    temp = np.array([[589.3667059623796,0.0,320.0],[0.0,589.3667059623796,240.0],[0.0,0.0,1.0]]) 
                    temp2= np.linalg.inv(temp)
                    threed=[]
                    for i in range(len(x)):
                        temp3=np.array([x[i],y[i],1])
                        threed.append(np.dot(temp2,temp3))
                    xline=[]
                    yline=[]
                    zline=[]

                    for i in range(len(threed)):
                        xline.append(threed[i][0])
                        yline.append(threed[i][1])
                        zline.append(threed[i][2])
                    print("x:",xline)
                    print("y:",yline) 
                    print("z:",zline)
                    ax.plot3D(xline, yline,zline)
                    ax.scatter(xline,yline, zline)
            name = src_point['part'] + dst_point['part']
            check_and_drawline(name)
            name = dst_point['part'] + src_point['part']
            check_and_drawline(name)
    plt.draw()
    plt.pause(0.0000000000001)
    return None

color_table = [(0,255,0), (255,0,0), (0,0,255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]


############################################pose estimation################################

def pose_estimation(body,img):
    center=(int((body['keypoints'][5]['position']['x']+body['keypoints'][6]['position']['x'])//2),int((body['keypoints'][5]['position']['y']+body['keypoints'][6]['position']['y'])//2))
    cv2.circle(img,center,10,(255,255,255),-1)
    row='center'
    column='center'
    if(center[0]>430 and center[0]<860):
        row='center'
    elif(center[0]<430):
        row='right'
    elif(center[0]>860):
        row='left'
    if(center[1]>240 and center[1]<480):
        column='center'
    elif(center[1]<240):
        column='up'
    elif(center[1]>480):
        column='bottom'
    print("row:",row)
    print("column:",column)
    cv2.putText(img, row, (50,430), cv2.FONT_HERSHEY_SIMPLEX,3, (0,255,255), 5, cv2.LINE_AA) 
    cv2.putText(img, column, (50,490), cv2.FONT_HERSHEY_SIMPLEX,3, (0,255,255), 5, cv2.LINE_AA)
    




########################################## Tensorflow Lite interpreter #########################
interpreter = tf.lite.Interpreter(model_path='posenet.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
# ####################### Use help(tf.lite.Interpreter) in case you need some help #####################
# print(f'These are input details : {input_details}')
# print(f'These are output details : {output_details}')
# These are input details : [{'name': 'sub_2', 'index': 93, 'shape': array([  1, 257, 257,   3]), 'dtype': <class 'numpy.float32'>, 'quantization': (0.0, 0)}]
# These are output details : [{'name': 'MobilenetV1/heatmap_2/BiasAdd', 'index': 87, 'shape': array([ 1,  9,  9, 17]), 'dtype': <class 'numpy.float32'>, 'quantization': (0.0, 0)}, {'name': 'MobilenetV1/offset_2/BiasAdd', 'index': 90, 'shape': array([ 1,  9,  9, 34]), 'dtype': <class 'numpy.float32'>, 'quantization': (0.0, 0)}, {'name': 'MobilenetV1/displacement_fwd_2/BiasAdd', 'index': 84, 'shape': array([ 1,  9,  9, 32]), 'dtype': <class 'numpy.float32'>, 'quantization': (0.0, 0)}, {'name': 'MobilenetV1/displacement_bwd_2/BiasAdd', 'index': 81, 'shape': array([ 1,  9,  9, 32]), 'dtype': <class 'numpy.float32'>, 'quantization': (0.0, 0)}]
############ With the above info we understand that we get heatmaps(scores) , offsets(locations of keypoints) and their displacement gradients/vectors #####################################

################ Defining details of capture window ###################

frame_count = 0
cv2.namedWindow("test")
cv2.resizeWindow('test',cam_width,cam_height)
cap = cv2.VideoCapture(cam_id)
cap.set(3, cam_width)
cap.set(4, cam_height)
img_counter = 0

counter = 0

text = "Be at the center and away from light source"
nearCenter = (500,300)
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.7
fontColor = (41, 67, 240)
lineType = 2


fig = plt.figure()
while True:
    ret, frame = cap.read()
    cv2.line(frame,(430,0),(430,720),(255,255,255),5)
    cv2.line(frame,(860,0),(860,720),(255,255,255),5)
    cv2.line(frame,(0,240),(1290,240),(255,255,255),5)
    cv2.line(frame,(0,480),(1290,480),(255,255,255),5)
    if not ret:
        print("failed to grab frame")
        break
    input_data = cv2.resize(frame,(input_details[0]['shape'][1],input_details[0]['shape'][2]),interpolation=cv2.INTER_CUBIC).astype(np.float32)
    input_data = cv2.cvtColor(input_data,cv2.COLOR_BGR2RGB).astype(np.float32)
    input_data = input_data * (2.0 / 255.0) - 1.0
    input_data = np.expand_dims(input_data, axis=0)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    heatmaps_result = np.array(interpreter.get_tensor(output_details[0]['index']))
    offsets_result = np.array(interpreter.get_tensor(output_details[1]['index']))
    # displacementFwd_result = np.array(interpreter.get_tensor(output_details[2]['index']))
    # displacementBwd_result = np.array(interpreter.get_tensor(output_details[3]['index']))
    heatmaps_result = np.squeeze(heatmaps_result)
    offsets_result = np.squeeze(offsets_result)
    # displacementFwd_result = np.squeeze(displacementFwd_result)
    # displacementBwd_result = np.squeeze(displacementBwd_result)
    poses = decode_single_pose(heatmaps_result, offsets_result)
    
    for i in range(len(poses)):
        
        # t1 = threading.Thread(target=drawmatplotlib, args=(poses[i],fig,)) 
        # t1.start()
        # drawmatplotlib(poses[i],fig)
        pose_estimation(poses[i],frame)
        
        
        if poses[i]['score'] > 0.1:
            color = color_table[i]
            drawKeypoints(poses[i], frame, color)
            drawSkeleton(poses[i],frame)
        else:
            counter += 1
            if counter>=5:
                cv2.putText(frame,text,nearCenter,font,fontScale,fontColor,lineType)
                if counter >=8 :
                    counter = 0  
    cv2.imshow("test", frame)
    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        cap.release()
        cv2.destroyAllWindows()
        print("Escape hit, closing...")
        break

