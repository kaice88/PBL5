import subprocess
import os
import pickle
import time

from ExtractFeature import *

def predict(forest, input_path, output_path):

    # Define the command to run the Python file with arguments
    current_millis = time.time() * 1000

    current_directory = os.path.dirname(__file__)
    alphapose_script = os.path.join(
                            current_directory, "AlphaPose/scripts/demo_inference.py")
    command = ['python', alphapose_script, '--image', input_path, '--outdir', output_path]

    # Run the command
    result = subprocess.run(command, capture_output=True, text=True)
    print("Extract duration:", round(time.time() * 1000) - current_millis)

    json_result_path = os.path.join(
                            output_path, "alphapose-results.json")
    keypoints_result_path = os.path.join(
                            output_path, "keypoints.csv")
    features_result_path = os.path.join(
                            output_path, "features.csv")
    #
    extract_keypoints(json_result_path, keypoints_result_path)

    extract_features(keypoints_result_path, features_result_path)

    features_array = pd.read_csv(features_result_path,header=None).values
    y_pred = forest.predict(features_array)
    return y_pred
