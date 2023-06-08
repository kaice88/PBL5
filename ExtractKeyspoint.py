import json


def extract_keypoints(json_file_path):
    # Open the JSON file
    with open(json_file_path) as f:
        data = json.load(f)

    # Extract the keypoints array
    keypoints_list = []

    for item in data:
        keypoints_list.append((item['image_id'], item['keypoints']))
    return keypoints_list


A = extract_keypoints(
    "C:/Users/Lenovo/Downloads/Processed/valid/alphapose/alphapose-results.json")

B = []
C = []
for x in A:
    C.append(x[0])

for y in C:
    if C.count(y) > 1 and y not in B:
        B.append(y)

new_A = [x for x in A if x[0] not in B]
new_A_name = [x[0] for x in new_A]
new_A_kp = [x[1] for x in new_A]
print(new_A_name)
boxes = [0, 0.5, 0.5, 1, 1]

A_str = []
for i in range(len(new_A_kp)):
    new_arr = []
    for j in range(len(new_A_kp[i])):
        if (j+1) % 3 == 0:
            continue
        else:
            new_arr.append(new_A_kp[i][j])

    for k in range(len(new_arr)):
        if k % 2 == 0:
            new_arr[k] = new_arr[k]/192
        else:
            new_arr[k] = new_arr[k]/256

    new_arr = boxes + new_arr
    s = " ".join(str(x) for x in new_arr)
    A_str.append(s)


for i in range(len(A_str)):
    file_path = "C:/Users/Lenovo/Downloads/Processed/valid/Labels/" + \
        new_A_name[i]
    new_filename = file_path.replace(".jpg", ".txt")
    with open(new_filename, "w") as file:
        file.write(A_str[i])
