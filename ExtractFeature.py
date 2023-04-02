
import json
import csv
import math
import pandas as pd
import os
from sklearn.model_selection import train_test_split


current_directory = os.getcwd()


def checkvalid(arr):
    x_keypoints = [x for idx, x in enumerate(
        arr) if idx % 3 == 0]

    y_keypoints = [y for idx, y in enumerate(
        arr) if idx % 3 == 1]
    if any(i < 0 for i in arr) or any(i > 256 for i in y_keypoints) or any(i > 192 for i in x_keypoints):
        return False
    else:
        return True


def extract_keypoints(json_file_path, keypoints_file_path):
    # Open the JSON file
    with open(json_file_path) as f:
        data = json.load(f)
    print(data)


    # Extract the keypoints array
    keypoints_list = []
    keypoints_list2 = []
    dis_list = []
    image_id_list = []
    for item in data:
        x1 = item['keypoints'][0]
        y1 = item['keypoints'][1]
        x2 = item['keypoints'][48]
        y2 = item['keypoints'][49]
        d = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if item['image_id'] not in image_id_list:
            image_id_list.append(item['image_id'])
            dis_list.append(d)
            keypoints_list.append(item['keypoints'])
        elif d > dis_list[len(dis_list)-1]:
            dis_list.pop(-1)
            keypoints_list.pop(-1)
            dis_list.append(d)
            keypoints_list.append(item['keypoints'])
    print(keypoints_list)
    for x in keypoints_list:
        if checkvalid(x):
            keypoints_list2.append(x)
    print(keypoints_list2)
    with open(keypoints_file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(keypoints_list2)


def angle_between_vectors(u, v, range_0_360=False):
    dot_product = u[0] * v[0] + u[1] * v[1]
    mag_u = math.sqrt(u[0]**2 + u[1]**2)
    mag_v = math.sqrt(v[0]**2 + v[1]**2)
    angle = math.atan2(u[0]*v[1] - u[1]*v[0], dot_product) * 180 / math.pi
    if range_0_360:
        angle = angle % 360
    else:
        if angle > 180:
            angle -= 360
    return angle


def extract_features(keypoints_file_path, features_file_path1):
    keypoints_list = pd.read_csv(
        keypoints_file_path,header=None).values

    features_list = []

    # tính góc (A(4):tai M(6): vai, B(12):hông)
    for k in keypoints_list:

        A = [k[3*3], k[3*3+1]]
        M = [k[5*3], k[5*3+1]]
        B = [k[11*3], k[11*3+1]]
        N = [k[13*3], k[13*3+1]]
        C = [k[15*3], k[15*3+1]]
        D = [k[12*3], k[12*3+1]]
        O = [k[14*3], k[14*3+1]]
        E = [k[16*3], k[16*3+1]]

        MA = [A[0]-M[0], A[1]-M[1]]
        MB = [B[0]-M[0], B[1]-M[1]]

        NB = [B[0]-N[0], B[1]-N[1]]
        NC = [C[0]-N[0], C[1]-N[1]]

        OD = [D[0]-O[0], D[1]-O[1]]
        OE = [E[0]-O[0], E[1]-O[1]]

        angle1 = angle_between_vectors(MA, MB, True)
        angle2 = angle_between_vectors(NB, NC, True)
        angle3 = angle_between_vectors(OD, OE, True)

        ratio = (math.sqrt((B[0] - M[0])**2 + (B[1] - M[1])**2)) / \
            (math.sqrt((C[0] - A[0])**2 + (C[1] - A[1])**2))
        features_list.append(
            (angle1, ratio, angle2, angle3, abs(angle2-angle3)))
    print(features_list)
    with open(features_file_path1, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(features_list)


def data_split(features_file_path):
    # Load the dataset
    dataset = pd.read_csv(
        features_file_path, header=None)

    # Shuffle the dataset
    dataset = dataset.sample(frac=1).reset_index(drop=True)

    # Split into train and test sets
    train, test = train_test_split(dataset, test_size=0.2)

    # Split the train set into train and validation sets
    train, validation = train_test_split(train, test_size=0.2)
    return train.values, test.values, validation.values


def find_files(name, path):
    result = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file == name:
                result.append(os.path.join(root, file))

    return result


if __name__ == "__main__":
    json_path = find_files(
        "alphapose-results.json", os.path.join(
            current_directory, "Result2/256x192"))
    print(json_path)


    for i in json_path:
        directory = os.path.dirname(i)
        keypoints_file_path = os.path.join(directory, "keypoints.csv")
        features_file_path = os.path.join(directory, "features.csv")
        train_file_path = os.path.join(directory, "train.csv")
        test_file_path = os.path.join(directory, "test.csv")
        valid_file_path = os.path.join(directory, "valid.csv")
        extract_keypoints(i, keypoints_file_path)
        extract_features(keypoints_file_path, features_file_path)

    output_path = os.path.join(current_directory, "features.csv")
    features_path = find_files(
        "features.csv", os.path.join(
            current_directory, "Result2/256x192"))

    # for i in keypoints_path:
    #     # directory = os.path.dirname(i)
    #     # Create an empty output CSV file

    with open(output_path, "w", newline="") as output_file:
        writer = csv.writer(output_file)

        # Loop over each CSV file and append its contents to the output CSV file
        for file_path in features_path:
            with open(file_path, "r") as input_file:
                reader = csv.reader(input_file)


                # Write the remaining rows to the output file
                for row in reader:
                    writer.writerow(row)
