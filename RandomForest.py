from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


def trainModel(data, label):
    X = data
    y = label
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    dt = DecisionTreeClassifier()

    # create a decision tree classifier
    dt = DecisionTreeClassifier()

    # train the classifier on the training data
    dt.fit(X_train, y_train)

    # make predictions on the test data
    y_pred = dt.predict(X_test)

    # evaluate the accuracy of the classifier
    accuracy = dt.score(X_test, y_test)
    print(y_test)
    print(y_pred)
    print("Accuracy: ", accuracy)


# Doc du lieu
CR_dataframe = pd.read_csv("CR.txt").iloc[:, 1:].values
WR_dataframe = pd.read_csv("WR.txt").iloc[:, 1:].values
label_vector = np.concatenate((np.full((30), 0), np.full((30), 1)))
data = np.concatenate((CR_dataframe, WR_dataframe))


for i in range(10):
    trainModel(data, label_vector)
