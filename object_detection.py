import os

import torch
import torchvision.models.detection
from PIL import Image
from torchvision import transforms as T
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights


def process_image(image, model, resized_height = 256, resized_width =192):
    img = T.ToTensor()(image)
    img = T.functional.adjust_brightness(img, 1.5)
    img = T.functional.adjust_contrast(img, 1.5)
    img = T.Grayscale()(img)
    with torch.no_grad():
        pred = model([img])

    person_indexes = (pred[0]["labels"] == 1)
    bounding_boxes = pred[0]["boxes"][person_indexes]
    S_max = 0
    bounding_box = None
    for bd in bounding_boxes:
        if (bd[2] - bd[0]) * (bd[3] - bd[1]) > S_max:
            S_max = (bd[2] - bd[0]) * (bd[3] - bd[1])
            bounding_box = bd
    x_min = int(bounding_box[0])
    y_min = int(bounding_box[1])
    x_max = int(bounding_box[2])
    y_max = int(bounding_box[3])
    height = y_max - y_min
    width = x_max - x_min
    top = y_min
    left = x_min

    img = T.functional.crop(image, top, left, height, width)
    img = T.Resize((resized_height, resized_width))(img)
    return T.ToTensor()(img)


model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
model.eval()
