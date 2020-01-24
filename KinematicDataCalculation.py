import math

# 平行移動
def Translation(center_point, moved_center_point, moved_keypoint):
    moving_distance = moved_center_point - center_point
    trans_key_point = moved_keypoint - moving_distance
    return trans_key_point

# 角度を求める
def Angle(center_point, key_point, trans_key_point):
    a_vector = key_point - center_point
    b_vector = trans_key_point - center_point
    naiseki = sum(a_vector * b_vector)
    a_vector_ex = (a_vector[0] * a_vector[0]) + (a_vector[1] * a_vector[1])
    b_vector_ex = (b_vector[0] * b_vector[0]) + (b_vector[1] * b_vector[1])
    cos = naiseki / math.sqrt(a_vector_ex * b_vector_ex)
    try:
        angle = math.degrees(math.acos(cos))
        return angle
    except ValueError:
        cos = 1.0
        angle = math.degrees(math.acos(cos))
        return angle

# 移動量
def Travel(key_point, moved_key_point):
    dist = math.sqrt((moved_key_point[0] - key_point[0]) ** 2 + (moved_key_point[1] - key_point[1]) ** 2)
    return dist


class Calculation:
    FPS = 60

    # 角速度を計算する
    def AngularVelocity(self, center_point, moved_center_point, key_point, moved_key_point, skip):
        time = skip / self.FPS
        trans_key_point = Translation(center_point, moved_center_point, moved_key_point)
        angle = Angle(center_point, key_point, trans_key_point)
        return angle / time

    # 速度を計算する
    def Speed(self, dist, skip):
        time = skip / self.FPS
        return dist / time
