import math

# 平行移動
def Translation(center_point, moved_center_point, moved_keypoint):
    moving_distance = moved_center_point - center_point
    trans_keypoint = moved_keypoint - moving_distance
    return trans_keypoint

# 角度を求める
def Angle(center_point, keypoint, trans_keypoint):
    a_vector = keypoint - center_point
    b_vector = trans_keypoint - center_point
    naiseki = sum(a_vector * b_vector)
    a_vector_ex = (a_vector[0] * a_vector[0]) + (a_vector[1] * a_vector[1])
    b_vector_ex = (b_vector[0] * b_vector[0]) + (b_vector[1] * b_vector[1])
    cos = naiseki / math.sqrt(a_vector_ex * b_vector_ex)
    angle = math.degrees(math.acos(cos))
    return angle

# 移動量
def Travel(keypoint, moved_keypoint):
    dist = math.sqrt((moved_keypoint[0]-keypoint[0])**2 + (moved_keypoint[1]-keypoint[1])**2)
    return dist


class Calculation:
    FPS = 60

    def AngularVelocity(self, center_point, moved_center_point, keypoint, moved_keypoint, skip):
        time = skip / self.FPS
        trans_keypoint = Translation(center_point, moved_center_point, moved_keypoint)
        angle = Angle(center_point, keypoint, trans_keypoint)
        return angle / time

    def Speed(self, dist, skip):
        time = skip / self.FPS
        return dist / time
