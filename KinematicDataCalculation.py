import math
import numpy as np

FPS = 60
center_point = np.array([0, 0])
moved_center_point = np.array([0, 0])
joint = np.array([0, 1])
moved_joint = np.array([1, 0])


# 角速度を求める
def AngularVelocity(center_point, moved_center_point, joint, moved_joint):
    moving_distance = moved_center_point - center_point
    trans_joint = moved_joint - moving_distance
    angle = Angle(center_point, joint, trans_joint)
    print(angle / fps_time(skip_fps=10))
    return angle / fps_time(skip_fps=10)


# 角度を求める
def Angle(center_point, joint, trans_joint):
    a_vector = joint - center_point
    b_vector = trans_joint - center_point
    naiseki = sum(a_vector * b_vector)
    a_vector_ex = (a_vector[0] * a_vector[0]) + (a_vector[1] * a_vector[1])
    b_vector_ex = (b_vector[0] * b_vector[0]) + (b_vector[1] * b_vector[1])
    cos = naiseki / math.sqrt(a_vector_ex * b_vector_ex)
    angle = math.degrees(math.acos(cos))
    return angle


def fps_time(skip_fps):
    return skip_fps / FPS


AngularVelocity(center_point, moved_center_point, joint, moved_joint)
