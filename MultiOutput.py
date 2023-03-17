
import numpy as np
import pandas as pd
import math
import os
from sklearn.utils import shuffle
from sklearn.multioutput import MultiOutputClassifier
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


current_directory = os.getcwd()


X_train = np.empty((0, 5))
X_test = np.empty((0, 5))
X_valid = np.empty((0, 5))
y_train = np.empty((0, 3))
y_test = np.empty((0, 3))
y_valid = np.empty((0, 3))


def find_files(name, path):
    result = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file == name:
                result.append(os.path.join(root, file))

    return result


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


def data_label(file_name, num_of_samples):
    y0 = np.ones((num_of_samples)) if file_name[0] == 'T' else np.zeros(
        (num_of_samples))
    y1 = np.ones((num_of_samples)) if file_name[1] == 'T' else np.zeros(
        (num_of_samples))
    y2 = np.ones((num_of_samples)) if file_name[2] == 'T' else np.zeros(
        (num_of_samples))
    y = np.vstack((y0, y1, y2)).T
    return y


features_files_path1 = find_files(
    "features.csv", os.path.join(
        current_directory, "Result1/256x192"))
features_files_path2 = find_files(
    "features.csv", os.path.join(
        current_directory, "Result2/256x192"))

features_files_path = [features_files_path1 + features_files_path2]


for i in features_files_path:
    for file in i:
        train, test, validation = data_split(file)

        r_train, c = train.shape
        r_test, c = test.shape
        r_valid, c = validation.shape

        X_train = np.concatenate((X_train, train), axis=0)
        X_test = np.concatenate((X_test, test), axis=0)
        X_valid = np.concatenate((X_valid, validation), axis=0)

        file_name = file.split('\\')[-2]
        y_train = np.concatenate(
            (y_train, data_label(file_name, r_train)), axis=0)
        y_test = np.concatenate(
            (y_test,  data_label(file_name, r_test)), axis=0)
        y_valid = np.concatenate(
            (y_valid,  data_label(file_name, r_valid)), axis=0)


def multi_output(X_train, y_train, X, y):
    forest = RandomForestClassifier(random_state=1)
    forest = MultiOutputClassifier(forest)
    forest.fit(X_train, y_train)
    y_pred = forest.predict(X)
    # Tính độ chính xác
    accuracy = accuracy_score(y, y_pred)
    accuracy1 = accuracy_score(y[:, 0], y_pred[:, 0])
    accuracy2 = accuracy_score(y[:, 1], y_pred[:, 1])
    accuracy3 = accuracy_score(y[:, 2], y_pred[:, 2])
    print("General: ", accuracy)
    print("Head: ", accuracy1)
    print("Back: ", accuracy2)
    print("Leg: ", accuracy3)
    print("-------------------------------------------")


print("Train: ")
multi_output(X_train, y_train, X_train, y_train)
print("Valid: ")
multi_output(X_train, y_train, X_valid, y_valid)
print("Test: ")
multi_output(X_train, y_train, X_test, y_test)
