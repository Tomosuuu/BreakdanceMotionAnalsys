import math
import numpy as np

class Calculation:
    FPS = 60
    # skip_fps = 60
    # center_point = np.array([0, 0])
    # moved_center_point = np.array([0, 0])
    # joint = np.array([0, 1])
    # moved_joint = np.array([1, 0])

    # def __init__(self):
    #     self.AngularVelocity(self.center_point, self.moved_center_point, self.joint, self.moved_joint)

    def AngularVelocity(self, center_point, moved_center_point, joint, moved_joint, skip):
        time = skip / self.FPS
        trans_joint = self.Translation(center_point, moved_center_point, moved_joint)
        angle = self.Angle(center_point, joint, trans_joint)
        print(angle / time)

    # 平行移動
    def Translation(self, center_point, moved_center_point, moved_joint):
        moving_distance = moved_center_point - center_point
        trans_joint = moved_joint - moving_distance
        return trans_joint

    # 角度を求める
    def Angle(self, center_point, joint, trans_joint):
        a_vector = joint - center_point
        b_vector = trans_joint - center_point
        naiseki = sum(a_vector * b_vector)
        a_vector_ex = (a_vector[0] * a_vector[0]) + (a_vector[1] * a_vector[1])
        b_vector_ex = (b_vector[0] * b_vector[0]) + (b_vector[1] * b_vector[1])
        cos = naiseki / math.sqrt(a_vector_ex * b_vector_ex)
        angle = math.degrees(math.acos(cos))
        return angle


Calculation()
