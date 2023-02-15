import mediapipe as mp
import numpy as np
import cv2
from numpy.linalg import norm

output_dir = "C:/Users/Lenovo/Desktop/KI6/PBL5"


# khởi tạo thư viện mediapipe
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils  # vẽ khung xương trên ảnh

# lưu thông số từng điểm trên khung xương
lm_list1 = np.empty((2, 15, 2))
lm_list2 = np.empty((2, 15, 2))


def make_landmark_timestep1(results, img):
    c_lm = np.empty((15, 2))
    h, w, c = img.shape
    for id, lm in enumerate(results.pose_landmarks.landmark):
        if (id < 15):
            cx, cy = int(lm.x * w), int(lm.y * h)
            c_lm[id, 0] = cx
            c_lm[id, 1] = cy
    print(c_lm)
    return c_lm


def make_landmark_timestep2(results, img):
    c_lm = np.empty((15, 2))
    for id, lm in enumerate(results.pose_landmarks.landmark):
        if (id < 15):
            c_lm[id, 0] = lm.x
            c_lm[id, 1] = lm.y
    print(c_lm)
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


def processImage(id, image_path1, output_dir, image_path2):
    image = cv2.flip(cv2.imread(image_path1), 1)
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.pose_landmarks:  # nếu phát hiện được khung xương ảnh
        # ghi nhận thông số khung xương
        lm1 = make_landmark_timestep1(results, image)
        lm2 = make_landmark_timestep2(results, image)
        lm_list1[id, :, :] = lm1
        lm_list2[id, :, :] = lm2
    # vẽ lên ảnh
        image1 = draw_landmark_on_image(mpDraw, image, results)
    # image2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        print(f"{output_dir}/{image_path2.split('/')[-1]}")
        cv2.imwrite(
            f"{output_dir}/{image_path2.split('/')[-1]}", image1)


def cosineSimilarity(lm1, lm2):
    A = lm1
    B = lm2
    cosine = np.sum(A*B, axis=1)/(norm(A, axis=1)*norm(B, axis=1))
    print("Cosine Similarity:\n", cosine)


processImage(0, "imageQC.jpg", output_dir, "imageLM1.jpg")
processImage(1, "imageTH.jpg", output_dir, "imageLM2.jpg")
cosineSimilarity(lm_list1[0, :, :], lm_list1[1, :, :])
cosineSimilarity(lm_list2[0, :, :], lm_list2[1, :, :])
