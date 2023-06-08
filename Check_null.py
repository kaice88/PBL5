import os

# Thay đổi đường dẫn thư mục tại đây
folder_path = "D:/YOLO/runs/pose/predict17/labels"
i = 0
for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        file_path = os.path.join(folder_path, filename)
        if os.path.getsize(file_path) == 0:
            os.remove(file_path)
            print(f"Đã xóa tệp tin trống: {filename}")
            i = i+1
            print(i)
