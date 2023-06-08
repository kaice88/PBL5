import pickle

import numpy as np
from pandas import DataFrame
from sklearn.metrics import ConfusionMatrixDisplay
from ultralytics import YOLO

import ObjectDetection.yolov8_od as od
from model.extract_feature import get_features
from model.predict import predict
# keypoint_model = YOLO('M:/HK6/PBL5/Repos/FastAPIServer/yolov8-pose-224.pt')
# keypoint_model = YOLO('yolov8x-pose.pt')
keypoint_model = YOLO('M:/HK6/PBL5/Repos/FastAPIServer/weights/yolov8_pose_13_5.pt')

source_folder = "D:/Dataset/ORIGINAL/TYPE1_DONE_CHI_154/100_TESTTING"
child_folders =["010"]
# source_folder = "D:/Dataset/New/ForwardedHead"
target_folder = "D:/Dataset/ORIGINAL/TYPE1_DONE_CHI_154/100_TESTTING/Processed"
with open('M:/HK6/PBL5/Repos/FastAPIServer/Sitting-Posture-Corrector-ML/model/head_forest.pkl', 'rb') as f:
    head_forest = pickle.load(f)
with open('M:/HK6/PBL5/Repos/FastAPIServer/Sitting-Posture-Corrector-ML/model/back_forest.pkl', 'rb') as f:
    back_forest = pickle.load(f)
with open('M:/HK6/PBL5/Repos/FastAPIServer/Sitting-Posture-Corrector-ML/model/leg_forest.pkl', 'rb') as f:
    leg_forest = pickle.load(f)
import os
head_accuracy = 0
back_accuracy = 0
leg_accuracy = 0
total = 0
result_table = DataFrame(columns=['folder', 'head', 'back', 'leg','wrong_images'])
debug_table = DataFrame(columns=['folder', 'real','image_name', 'head_angle', 'back_angle', 'leg_angle1', 'leg_angle2', 'abs'])
from PIL import Image
# Initialize confusion matrices for head, back, and leg
head_cm = np.zeros((2, 2), dtype=int)
back_cm = np.zeros((3, 3), dtype=int)
leg_cm = np.zeros((2, 2), dtype=int)
for folder in child_folders:
    fd = source_folder + "/" + folder
    head_folder = 0
    back_folder = 0
    leg_folder = 0
    for f in os.listdir(fd):
        try:
            image = Image.open(os.path.join(fd, f))
            model = YOLO('M:/HK6/PBL5/Repos/FastAPIServer/Sitting-Posture-Corrector-ML/ObjectDetection/runs/detect/train7/weights/best.pt')
            preprocessed_image = od.process_image(image, model)
            # For debugging
            preprocessed_image.save("preprocessed_image.jpg")
            # Step 2: Feature extraction and predict
            result = predict(head_forest, back_forest, leg_forest, keypoint_model, preprocessed_image)
            head_expected = folder[0]
            back_expected = folder[1]
            leg_expected = folder[2]

            if int(head_expected) == result[0][0]:
                head_accuracy+= 1
                head_folder+= 1
            if int(back_expected) == result[1][0]:
                back_accuracy+= 1
                back_folder +=1
            if int(leg_expected) == result[2][0]:
                leg_accuracy+= 1
                leg_folder +=1
            if not (int(head_expected) == result[0][0] and int(back_expected) == result[1][0] and int(leg_expected) == result[2][0]):
                keypoints = keypoint_model.predict(preprocessed_image, save=True)[0].keypoints[0].tolist()
                # Step 2: Extract features from keypoints
                features_array = get_features(keypoints)
                debug_table.loc[len(debug_table)] = {'folder': folder, 'real': str(result[0][0]) + str(result[1][0]) + str(result[2][0]), 'image_name': f, 'head_angle': features_array[0], 'back_angle': features_array[1], 'leg_angle1': features_array[2], 'leg_angle2': features_array[3], 'abs': features_array[4]}
            # print("Expected: " + folder + ", real: " + str(result[0][0]) + str(result[1][0]) + str(result[2][0]) + " for file " + f)
            head_cm[int(head_expected), result[0][0]] += 1
            back_cm[int(back_expected), result[1][0]] += 1
            leg_cm[int(leg_expected), result[2][0]] += 1
        except Exception as e:
            print(e)
            print(os.path.join(fd, f))
    result_table.loc[len(result_table)] = {
        "folder": folder,
        "head": (head_folder)/ len(os.listdir(source_folder + "/" + folder)),
        "back": (back_folder)/ len(os.listdir(source_folder + "/" + folder)),
        "leg": (leg_folder)/ len(os.listdir(source_folder + "/" + folder))
    }
    print("Head: " + str(head_folder) + "/" + str(len(os.listdir(source_folder + "/" + folder))))
    print("Back: " + str(back_folder) + "/" + str(len(os.listdir(source_folder + "/" + folder))))
    print("Leg: " + str(leg_folder) + "/" + str(len(os.listdir(source_folder + "/" + folder))))

print(result_table)
debug_table.to_csv("debug_table.csv")
# print(total)
print("Accuracy: ", head_accuracy/139, back_accuracy/139, leg_accuracy/139)
print(head_accuracy, back_accuracy, leg_accuracy)
ConfusionMatrixDisplay(head_cm, display_labels=[0, 1]).plot()
ConfusionMatrixDisplay(back_cm, display_labels=[1,2, 3]).plot()
ConfusionMatrixDisplay(leg_cm, display_labels=[0, 1]).plot()