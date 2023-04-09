import os
from ExtractFeature import *
import numpy as np
import matplotlib.pyplot as plt

current_directory = os.getcwd()

features_path = find_files(
    "features.csv", os.path.join(
        current_directory, "Result1/256x192"))
print(features_path)
X = []


def handling_feature(folder_name, features_file_path):
    # data = np.genfromtxt(features_file_path, delimiter=',')
    with open(features_file_path, 'r') as file:
        reader = csv.reader(file)
        data = [row for row in reader]

    result = [[x[0], folder_name[0], x[1]] for x in data]
    X.extend(result)


for i in features_path:
    parent_folder_name = os.path.basename(os.path.dirname(i))
    handling_feature(parent_folder_name, i)
print(X)
print(len(X))


# Define the data points


# Create two lists to store the x and y values
# x_values = list(range(len(X)))
# y_values = [d[2] for d in X]

# print(x_values, y_values)
# # Create two lists to store the colors for each X point
# colors = ['red' if d[1] == 'F' else 'blue' for d in X]

# # Create the scatter plot
# plt.scatter(x_values, y_values, c=colors)

# # Set the title and labels
# plt.title('Scatter Plot')
# plt.xlabel('Index')
# plt.ylabel('Angle')
# plt.show()

x_values = list(range(len(X)))
y_values = [float(d[2]) for d in X]

# Create two lists to store the colors for each X point
colors = ['red' if d[1] == 'F' else 'blue' for d in X]
labels = [d[0] for d in X]
for i, label in enumerate(labels):
    plt.annotate(label, (x_values[i], y_values[i]))

# Create the scatter plot
plt.scatter(x_values, y_values, c=colors)

# Set the title and labels
plt.title('Scatter Plot')
plt.xlabel('Index')
plt.ylabel('Angle')
plt.show()
# %%
