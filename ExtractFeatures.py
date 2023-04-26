import csv
import math
import os
import matplotlib.pyplot as plt
# str = "0 0.5 0.5 0.989583 1 0.598766 0.122288 0.620318 0.0851445 0.597729 0.0884052 0.740554 0.086699 0.741309 0.0854913 0.849263 0.26796 0.768917 0.265196 0.621398 0.463327 0.567804 0.454568 0.29054 0.442328 0.315768 0.443887 0.769182 0.668436 0.710865 0.630385 0.223487 0.705839 0.275299 0.671493 0.201536 0.968608 0.261856 0.931025"
# arr = str.split(" ")

# file csv
features_path = "C:/Users/Lenovo/Downloads/Test_Speech/features.csv"
labels_path = "C:/Users/Lenovo/Downloads/Test_Speech/labels.csv"


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


def handling_keypoints(keypoints):
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
labels = []

output_folder = "D:/YOLO/runs/pose/predict/labels"


def extract_features():
    file_names = os.listdir(output_folder)
    file_paths = [os.path.join(output_folder, file_name)
                  for file_name in file_names]
    keypoints = extract_keypoints(file_paths)
    for id, val in enumerate(keypoints):
        head, back, leg_right, leg_left = handling_keypoints(val)
        angel1, angle2, angle3, angle4 = feature(
            head, back, leg_right, leg_left)

        name = file_names[id].split("_")[0]
        features.append((name[0], name[1], name[2], angel1,
                        angle2, angle3, angle4, abs(angle3-angle4)))


def writeFile(path, data):
    with open(path, mode='w', newline='') as file:
        writer = csv.writer(file)
        for row in data:
            writer.writerow(row)


extract_features()
writeFile(features_path, features)
writeFile(labels_path, labels)
print(len(features))
print(len(labels))
# for id, val in enumerate(keypoints_list2):
#     head, back, leg_right, leg_left = handling_keypoints(val)
#     head, back, leg_right, leg_left = feature(head, back, leg_right, leg_left)
#     y_values.append(back)
#     labels.append(2)
#     names.append(file_names2[id].split('_')[1])

# for id, val in enumerate(keypoints_list3):
#     head, back, leg_right, leg_left = handling_keypoints(val)
#     head, back, leg_right, leg_left = feature(head, back, leg_right, leg_left)
#     y_values.append(back)
#     labels.append(3)
#     names.append(file_names3[id].split('_')[1])

# x_values = list(range(len(y_values)))
# colors = []
# for l in labels:
#     if l == 1:
#         colors.append('red')
#     elif l == 2:
#         colors.append('blue')
#     elif l == 3:
#         colors.append('green')

# for i, name in enumerate(names):
#     plt.annotate(name, (x_values[i], y_values[i]))

# plt.scatter(x_values, y_values, c=colors)

# # Set the title and labels
# plt.title('Scatter Plot')
# plt.xlabel('Index')
# plt.ylabel('Angle')
# plt.show()

# # head, back, leg_right, leg_left = handling_keypoints(arr)
# # print(feature(head, back, leg_right, leg_left))
