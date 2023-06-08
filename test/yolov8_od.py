from PIL import ImageOps, ImageEnhance


def process_image(input, model):
    img_data = ImageOps.exif_transpose(input)
    results_list = model.predict(img_data, conf= 0.4)[0]  #
    bounding_box = None
    max_width = 0
    for results in results_list:
        boxes = results.boxes
        for bd in boxes:
            if bd.cls[0] == 0:
                xyxy = bd.xyxy[0].tolist()
                if (xyxy[2] - xyxy[0]) > max_width:
                    max_width = xyxy[2] - xyxy[0]
                    bounding_box = xyxy
    predict_box = img_data.crop((bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3]))
    predict_box= predict_box.resize((192, 256))
    return predict_box