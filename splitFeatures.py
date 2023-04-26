import numpy as np
import pandas as pd


from sklearn.model_selection import train_test_split
# Tree Visualisation

from IPython.display import Image


# file csv
features_path = "C:/Users/Lenovo/Downloads/Test_Speech/features.csv"


def data_split(features_file_path):
    # Load the dataset
    dataset = pd.read_csv(
        features_file_path, header=None)

    # Shuffle the dataset
    dataset = dataset.sample(frac=1).reset_index(drop=True)

    # Split into train and test sets
    train, test = train_test_split(dataset, test_size=0.2)

    return train.values, test.values


train, test = data_split(features_path)
X_train = []
X_test = []
y_train = []
y_test = []
for i in train:
    X_train.append(i[3:])
    y_train.append(i[:3])
for i in test:
    X_test.append(i[3:])
    y_test.append(i[:3])
