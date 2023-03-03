import mediapipe as mp
import os
import numpy as np
import pandas as pd
import cv2
from numpy.linalg import norm
from BootsTrap import bootstrap


input_dir = ["C:/Users/Lenovo/Desktop/KI6/PBL5/Train/Demo/CR",
             "C:/Users/Lenovo/Desktop/KI6/PBL5/Train/Demo/WR1"]

output_dir = ["C:/Users/Lenovo/Desktop/KI6/PBL5/Train/Demo/RS0",
              "C:/Users/Lenovo/Desktop/KI6/PBL5/Train/Demo/RS1"]

# khởi tạo thư viện mediapipe
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils  # vẽ khung xương trên ảnh


# lưu thông số từng điểm trên khung xương
numberOfFiles = len(os.listdir(input_dir[0]))


# 2 class; số ảnh train mỗi class ;lấy 15 điểm mỗi điểm lấy x,y nên 15*2 = 30;
lm_list = np.empty((2, numberOfFiles, 30))
label_list = ["CR", "WR"]


def make_landmark_timestep(results, img):

    # lấy 15 điểm mỗi điểm lấy x,y
    c_lm = np.empty((15, 2), dtype="float")
    for id, lm in enumerate(results.pose_landmarks.landmark):
        if (id < 15):
            c_lm[id, 0] = lm.x
            c_lm[id, 1] = lm.y
            # c_lm[id, 2] = lm.z
    return c_lm.flatten()


def draw_landmark_on_image(mpDraw, img, results):
    # dùng thư viện vẽ của mediapipe
    # vẽ các đường nối
    mpDraw.draw_landmarks(
        img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    # vẽ các nút
    for id, lm in enumerate(results.pose_landmarks.landmark):
        if (id < 15):
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(img, (cx, cy), 7, (0, 0, 255), cv2.FILLED)
    return img


def processImage(label, id, image_path1, image_path2):

    image = cv2.imread(image_path1)
    with mpPose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=True,
        min_detection_confidence=0.5
    ) as pose:
        results = pose.process(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:  # nếu phát hiện được khung xương ảnh
            # ghi nhận thông số khung xương
            lm = make_landmark_timestep(results, image)
            lm_list[label, id, :] = lm
            # vẽ lên ảnh
            image1 = draw_landmark_on_image(mpDraw, image, results)
            # image2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(image_path2, image1)


#  chuẩn hóa phạm vi [0,1]
# def normalize(cm):
#     r, c = cm.shape
#     temp = np.empty((r, c), dtype="float")
#     maxX = max(cm[:, 0])
#     maxY = max(cm[:, 1])
#     for i in range(r):
#         temp[i, 0] = cm[i, 0]/maxX
#         temp[i, 1] = cm[i, 1]/maxY
#     return temp


# trích xuất vector đặc trưng và lưu vào file csv
for i, label in enumerate(label_list):
    for id, image in enumerate(os.listdir(input_dir[i])):
        processImage(
            i, id, f"{input_dir[i]}/{image}", f"{output_dir[i]}/{image}")
    # Lưu vào file csv
    df = pd.DataFrame(lm_list[i, :, :])
    df.to_csv(label + ".txt")


# đọc dữ liệu từ file csv

# CR_dataframe = pd.read_csv("CR.txt").iloc[:, 1:].values
# WR_dataframe = pd.read_csv("WR.txt").iloc[:, 1:].values

# sau khi lưu được vào file csv thì train data trong file RandomForest.py


# ----------------------------------------Random forest---------------------------------------------


# M = 10
# n = 40

# CR = bootstrap(lm_list[0, :, :], n, M)
# WR = bootstrap(lm_list[1, :, :], n, M)
# label_vector = np.concatenate((np.full((40), 0), np.full((40), 1)))

# for i in range(10):
#     data = np.concatenate((CR[i, :, :], WR[i, :, :]))
#     trainModel(data, label_vector)


# --------------------------------------------------------------------------------------------------
