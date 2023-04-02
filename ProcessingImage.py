import subprocess
import os
import pickle

from ExtractFeature import *




# Load the saved forest model
with open('trained_forest.pkl', 'rb') as f:
    forest = pickle.load(f)

print(forest)


# Define the command to run the Python file with arguments
current_directory = os.getcwd()
alphapose_script = os.path.join(
                        current_directory, "AlphaPose/scripts/demo_inference.py")
input_path = os.path.join(
                        current_directory, "TestImage/qc_img_6.jpg")
output_path = os.path.join(
                        current_directory, "TestImage/")
command = ['python', alphapose_script, '--image', input_path, '--outdir', output_path]

# Run the command
result = subprocess.run(command, capture_output=True, text=True)

json_result_path = os.path.join(
                        output_path, "alphapose-results.json")
# print(json_result_path)
keypoints_result_path = os.path.join(
                        output_path, "keypoints.csv")
features_result_path = os.path.join(
                        output_path, "features.csv")
#
extract_keypoints(json_result_path, keypoints_result_path)

extract_features(keypoints_result_path, features_result_path)

features_array = pd.read_csv(features_result_path,header=None).values

y_pred = forest.predict(features_array)
print(y_pred)
