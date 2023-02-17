import mediapipe as mp
import os
import numpy as np
import cv2
from numpy.linalg import norm
from createDecisionTree import createDecisionTree

input_dir = ["C:/Users/Lenovo/Desktop/KI6/PBL5/Train/CR",
             "C:/Users/Lenovo/Desktop/KI6/PBL5/Train/WR"]
output_dir = ["C:/Users/Lenovo/Desktop/KI6/PBL5/Train/RS0",
              "C:/Users/Lenovo/Desktop/KI6/PBL5/Train/RS1"]

# khởi tạo thư viện mediapipe
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils  # vẽ khung xương trên ảnh


# lưu thông số từng điểm trên khung xương
numberOfFiles = len(os.listdir(input_dir[0]))
lm_list = np.empty((2, numberOfFiles, 45))
label = [0, 1]


def make_landmark_timestep(results, img):
    c_lm = np.empty((15, 3), dtype="float")
    for id, lm in enumerate(results.pose_landmarks.landmark):
        if (id < 15):
            c_lm[id, 0] = lm.x
            c_lm[id, 1] = lm.y
            c_lm[id, 2] = lm.z
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
            cv2.imwrite(
                image_path2, image1)


# chuẩn hóa phạm vi [0,1]
def normalize(lm):
    r, c = lm.shape
    temp = np.empty((r, c), dtype="float")
    maxX = max(lm[:, 0])
    maxY = max(lm[:, 1])
    maxZ = max(abs(lm[:, 2]))
    for i in range(r):
        temp[i, 0] = lm[i, 0]/maxX
        temp[i, 1] = lm[i, 1]/maxY
        temp[i, 2] = lm[i, 2]/maxZ
    return temp


for i in label:
    for id, image in enumerate(os.listdir(input_dir[i])):
        processImage(
            i, id, f"{input_dir[i]}/{image}", f"{output_dir[i]}/{image}")


# print(lm_list[14, :, :])
# processImage(0, "C:/Users/Lenovo/Desktop/KI6/PBL5/IMG1.jpg",
#              "C:/Users/Lenovo/Desktop/KI6/PBL5/IMG2.jpg")
# processImage(0, "C:/Users/Lenovo/Desktop/KI6/PBL5/IMGTH2.jpg",
#              "C:/Users/Lenovo/Desktop/KI6/PBL5/IMGLM2.jpg")


# ----------------------------------------Random forest---------------------------------------------


M = 10
n = 40


def bootstrap(input, n, M):
    d2, d1 = np.shape(input)
    indexesOf1dArray = np.random.choice(
        np.arange(d2), size=(M, n), replace=True)  # 2d array

    results = np.empty((M, n, d1))
    for i in range(M):
        for j in range(n):
            indexOf1d = indexesOf1dArray[i, j]
            results[i, j, :] = input[indexOf1d, :]
    return results


CR = bootstrap(lm_list[0, :, :], n, M)
WR = bootstrap(lm_list[1, :, :], n, M)
label_list = np.concatenate((np.full((40), 0), np.full((40), 1)))

for i in range(10):
    data = np.concatenate((CR[i, :, :], WR[i, :, :]))
    createDecisionTree(data, label_list)


# --------------------------------------------------------------------------------------------------
