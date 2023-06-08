from model.extract_feature import *

def predict(head_forest, back_forest, leg_forest, keypoint_model,image):
    # Step 1: Extract keypoints
    keypoints = keypoint_model.predict(image, save=True)[0].keypoints[0].tolist()
    # Step 2: Extract features from keypoints
    features_array = get_features(keypoints)

    head_pred = head_forest.predict([[features_array[0]]])
    back_pred = back_forest.predict([[features_array[1]]])
    leg_pred = leg_forest.predict([features_array[2:4]])

    return [head_pred, back_pred, leg_pred]