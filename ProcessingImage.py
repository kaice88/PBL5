import subprocess
import os
import pickle
import  time

import numpy as np

from ExtractFeature import *
from AlphaPose.scripts.demo_inference import pose_model_1, pose_dataset_1 ,load_detector,args,set_args,detection





# Load the saved forest model
with open('trained_forest.pkl', 'rb') as f:
    forest = pickle.load(f)

# Define the command to run the Python file with arguments
current_directory = os.getcwd()
alphapose_script = os.path.join(
                        current_directory, "AlphaPose/scripts/demo_inference.py")
input_path = os.path.join(
                        current_directory, "TestImage/21.jpg")
output_path = os.path.join(
                        current_directory, "TestImage/")

set_args(input_path,output_path)

decorator_1 ,dec_worker_1= load_detector(args)

arr = detection(decorator_1,pose_model_1,pose_dataset_1)


keypoints_list = np.array(arr).flatten().tolist()
features_list = extract_features(keypoints_list)

y_pred = forest.predict(features_list)
print(y_pred)
