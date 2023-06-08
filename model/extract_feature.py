import math
import os

current_directory = os.getcwd()

def get_features(keypoints):
    #
    head_kp, back_kp, right_leg_kp, left_leg_kp = get_area_keypoints(keypoints)
    head_ag, back_ag, right_leg_ag, left_leg_ag = get_area_angles(head_kp, back_kp, right_leg_kp, left_leg_kp)
    return [head_ag, back_ag, right_leg_ag, left_leg_ag, abs(left_leg_ag - right_leg_ag)]
def get_area_keypoints(keypoints):
    # Resize to a 2d array
    # for i in range(0, len(keypoints), 3):
    #     new_keypoints.append((float(keypoints[i]) , float(keypoints[i + 1]), float(keypoints[i + 2])))
    # Calculate head & back keypoints (needed to calc the angles)
    # If the head is on the left side of the body
    if (keypoints[11][0] > keypoints[15][0]):
        head = [keypoints[3], keypoints[5], keypoints[11]]
        back = [keypoints[5], keypoints[11], keypoints[13]]
    # If the head is on the right side
    else:
        head = [keypoints[4], keypoints[6], keypoints[12]]
        back = [keypoints[6], keypoints[12], keypoints[14]]
    # Calculate leg keypoints
    right_leg = [keypoints[11], keypoints[13], keypoints[15]]
    left_leg = [keypoints[12], keypoints[14], keypoints[16]]
    return head, back, right_leg, left_leg


def angle_between_vectors_degrees(vector1, vector2):
    """
    Calculates the angle in degrees between two 2D vectors.

    Args:
        vector1 (tuple or list): A tuple or list representing the first vector with its x and y components.
        vector2 (tuple or list): A tuple or list representing the second vector with its x and y components.

    Returns:
        float: The angle between the two vectors in degrees.
    """
    # Extracting components of the vectors
    Ax, Ay = vector1
    Bx, By = vector2

    # Calculating dot product
    dot_product = Ax * Bx + Ay * By

    # Calculating magnitudes of the vectors
    magnitude_vector1 = math.sqrt(Ax ** 2 + Ay ** 2)
    magnitude_vector2 = math.sqrt(Bx ** 2 + By ** 2)

    # Calculating cosine of the angle
    cosine_theta = dot_product / (magnitude_vector1 * magnitude_vector2)

    # Calculating angle in degrees
    angle_degrees = math.degrees(math.acos(cosine_theta))

    return angle_degrees


def cal_angle(A, C, B):
    CA = [A[0]-C[0], A[1]-C[1]]
    CB = [B[0]-C[0], B[1]-C[1]]
    angle = angle_between_vectors_degrees(CA, CB)
    return angle


def get_area_angles(_head, _back, _right_leg, _left_leg):
    head_ag = cal_angle(_head[0], _head[1], _head[2])
    back_ag = cal_angle(_back[0], _back[1], _back[2])

    right_leg_ag = cal_angle(_right_leg[0], _right_leg[1], _right_leg[2])
    leg_lef_ag = cal_angle(_left_leg[0], _left_leg[1], _left_leg[2])

    return head_ag, back_ag, leg_lef_ag, right_leg_ag

