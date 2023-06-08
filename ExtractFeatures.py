import csv
import math
import os
import matplotlib.pyplot as plt

#
features_path = "C:/Users/Lenovo/Desktop/Processed/Features_Head.csv"
output_folder = "D:/YOLO/runs/pose/predict17/labels"


def extract_keypoints(_file_paths):
    keypoints = []
    for f in _file_paths:
        try:
            with open(f, 'r') as file:
                data = [line.rstrip() for line in file.readlines()]
                s = data[0]
                arr = s.split(" ")
                keypoints.append(arr)
        except FileNotFoundError:
            print("File not found")
    return keypoints


def handling_keypoints(arr):
    keypoints = []
    for i in range(5, len(arr), 2):
        keypoints.append((float(arr[i])*192, float(arr[i+1])*256))
    if (keypoints[11][0] > keypoints[15][0]):
        head = [keypoints[3], keypoints[5], keypoints[11]]
        # temp = (0, keypoints[13][1])
        back = [keypoints[5], keypoints[11], keypoints[13]]
    else:
        head = [keypoints[4], keypoints[6], keypoints[12]]
        back = [keypoints[6], keypoints[12], keypoints[14]]
    leg_right = [keypoints[11], keypoints[13], keypoints[15]]
    leg_left = [keypoints[12], keypoints[14], keypoints[16]]
    return head, back, leg_right, leg_left


def angle_between_vectors(u, v, range_0_360=False):
    dot_product = u[0] * v[0] + u[1] * v[1]
    angle = math.atan2(u[0]*v[1] - u[1]*v[0], dot_product) * 180 / math.pi
    if range_0_360:
        angle = angle % 360
    else:
        if angle > 180:
            angle -= 360
    return angle


def cal_angle(A, C, B):
    CA = [A[0]-C[0], A[1]-C[1]]
    CB = [B[0]-C[0], B[1]-C[1]]
    angle = angle_between_vectors(CA, CB, True)
    if (angle >= 180):
        angle = 360-angle
    return angle


def feature(_head, _back, _leg_right, _leg_left):
    angle_head = cal_angle(_head[0], _head[1], _head[2])
    angle_back = cal_angle(_back[0], _back[1], _back[2])

    angle_leg_right = cal_angle(_leg_right[0], _leg_right[1], _leg_right[2])
    angle_leg_left = cal_angle(_leg_left[0], _leg_left[1], _leg_left[2])

    return angle_head, angle_back, angle_leg_left, angle_leg_right


features = []


def extract_features():
    file_names = os.listdir(output_folder)
    file_paths = [os.path.join(output_folder, file_name)
                  for file_name in file_names]
    keypoints = extract_keypoints(file_paths)
    for id, val in enumerate(keypoints):
        head, back, leg_right, leg_left = handling_keypoints(val)
        angle1, angle2, angle3, angle4 = feature(
            head, back, leg_right, leg_left)

        name = file_names[id].split("_")[0]
        label = '1' if name == 'CH' else '0'
        # if name == 'CB':
        #     label = 1
        # elif name == 'LB':
        #     label = 2
        # elif name == 'LF':
        #     label = 3
        # features.append((label, angle2))
        features.append((label, angle1))


def writeFile(path, data):
    with open(path, mode='w', newline='') as file:
        writer = csv.writer(file)
        for row in data:
            writer.writerow(row)


extract_features()
writeFile(features_path, features)
