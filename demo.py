import mediapipe as mp
import os
import numpy as np
import cv2
from numpy.linalg import norm

output_dir1 = "C:/Users/Lenovo/Desktop/KI6/PBL5/Train/CR"
output_dir2 = "C:/Users/Lenovo/Desktop/KI6/PBL5/Train/RS"

# khởi tạo thư viện mediapipe
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils  # vẽ khung xương trên ảnh

# lưu thông số từng điểm trên khung xương
numberOfImage = len(os.listdir(output_dir1))
lm_list = np.empty((numberOfImage, 45))


def make_landmark_timestep(results, img):
    c_lm = np.empty((15, 3))

    for id, lm in enumerate(results.pose_landmarks.landmark):
        if (id < 15):
            c_lm[id, 0] = lm.x
            c_lm[id, 1] = lm.y
            c_lm[id, 2] = lm.z

    return c_lm


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


def processImage(id, image_path1, image_path2):

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
            print(lm)
            # lm_list[id, :] = lm
            # vẽ lên ảnh
            image1 = draw_landmark_on_image(mpDraw, image, results)
            # image2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(
                image_path2, cv2.flip(image1, 1))

# chuẩn hóa phạm vi [0,1]


def normalize(lm):
    r, c = lm.shape
    for i in range(r):
        lm[i,] = lm[i]


processImage(0, "C:/Users/Lenovo/Desktop/KI6/PBL5/IMGQC.jpg",
             "C:/Users/Lenovo/Desktop/KI6/PBL5/IMGLM2.jpg")
