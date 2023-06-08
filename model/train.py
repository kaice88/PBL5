import os
import pickle

import pandas as pd
import numpy as np

# Modelling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split

# Tree Visualisation
from sklearn.tree import export_graphviz
from IPython.display import display

import graphviz
#
# Tree Visualisation
#
#
current_directory = os.getcwd()

    # Export the first three decision trees from the forest
# #
#     for i in range(3):
#         tr = forest.estimators_[i]
#         dot_data = export_graphviz(tr,
#                                    feature_names=["A1", "R",
#                                                   "A2", "A3", "A2-A3"],
#                                    filled=True,
#                                    max_depth=2,
#                                    impurity=False,
#                                    proportion=True)
#         graph = graphviz.Source(dot_data)
#         Image(graph)
#
def train(X, y, pkl_file_path):
    X_train,  X_test,y_train, y_test = train_test_split(X,y,random_state=2, shuffle=True, stratify=y)
    forest = RandomForestClassifier(n_estimators=100, random_state=1)
    forest.fit(X_train, y_train)
    y_pred = forest.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: ", accuracy)
    print("-------------------------------------------")
    with open(pkl_file_path, 'wb') as f:
        pickle.dump(forest, f)

    for i in range(3):
        tree =forest.estimators_[i]
        dot_data = export_graphviz(tree,
                                   feature_names=['angle'],
                                   filled=True,
                                   max_depth=2,
                                   impurity=False,
                                   proportion=True)
        graph = graphviz.Source(dot_data, format='png')
        display(graph)
#
print("Test: ")


#
head_data = pd.read_csv("../data/Features_Head.csv")
y_head = head_data['label']
X_head = np.array(head_data['angle']).reshape(-1, 1)
train(X_head,y_head, 'head_forest.pkl')



# back_data = pd.read_csv("Features_Back.csv")
# y_back = back_data['label']
# X_back = np.array(back_data['back_angle']).reshape(-1, 1)
# train(X_back, y_back, 'back_forest.pkl')
# #
# leg_data = pd.read_csv("Features_Leg.csv")
# y_leg = leg_data['label'].values
# X_leg = leg_data[['leg1','leg2','diff']].values
# train(X_leg, y_leg, 'leg_forest.pkl')
# import pandas as pd
# import matplotlib.pyplot as plt
# head_data = pd.read_csv("../data/Features_Head.csv")
# # head_data[head_data[head_data['label'] == 0]['head_angle'] > 145]['head_angle'] = 145
# true_head = head_data[head_data['label'] == 1]['head_angle']
# wrong_head = head_data[head_data['label'] == 0]['head_angle']
# true_head.plot(kind='hist', bins = 5)
# wrong_head.plot(kind = 'hist', bins = 5)
# head_data.to_csv('Features_Head_Filter.csv')
