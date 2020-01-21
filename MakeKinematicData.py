from KinematicDataCalculation import Calculation, Translation, Angle, Travel
from MakeDataSet import DataSet, file_count
import pandas as pd
import itertools
import os
import glob


class KinematicData(Calculation):
    SKIP = 10
    KEY_POINTS_NAME = ['Nose',
                       'Heart',
                       'right_shoulder',
                       'right_elbow',
                       'right_wrist',
                       'left_shoulder',
                       'left_elbow',
                       'left_wrist',
                       'right_hip',
                       'right_knee',
                       'right_ankle',
                       'left_hip',
                       'left_knee',
                       'left_ankle',
                       'right_eye',
                       'left_eye',
                       'right_ear',
                       'left_ear']
    KEY_POINTS_PERM = list(itertools.permutations(KEY_POINTS_NAME, 2))
    CENTER_POINT = KEY_POINTS_NAME[1]
    KINEMATIC_DATA_PATH = "./Kinematic_Data"

    def __init__(self):
        kumiawase = self.Combination(self.CENTER_POINT)
        os.makedirs(self.KINEMATIC_DATA_PATH, exist_ok=True)
        count = file_count(DataSet.TRANS_DATA_PATH)
        for i in range(count):
            data_path = self.KINEMATIC_DATA_PATH + "/" + str(i).zfill(3)
            os.makedirs(data_path, exist_ok=True)
            dir_count = str(i).zfill(3)
            for a, b in kumiawase:
                kinematic_data = pd.DataFrame()
                kinematic_data['Angular_Velocity'] = self.calculate_angular_velocity(a, b, dir_count)
                # kinematic_data['Angle'] = self.calculate_angle(a, b, dir_count)
                kinematic_data.to_csv(data_path + "/" + a + " - " + b + ".csv")
        self.out_csv_file()

    # Heartとそれ以外のkey_pointの組み合わせを返す
    def Combination(self, center):
        Combination = []
        for i in self.KEY_POINTS_NAME:
            if i != center:
                Combination.append([center, i])
        return Combination

    # 未使用
    def one_key_point(self, keypoint_a, dir_count):
        pose = pd.read_csv(DataSet.TRANS_DATA_PATH + "/" + dir_count + "/" + str(keypoint_a) + '.csv', index_col=0)
        joi_a = pose.loc[pose.index % self.SKIP == 0, 'x':'y']
        A = joi_a.values

        return A

    # 2つのkey_pointを選択する
    def select_key_points(self, keypoint_a, keypoint_b, dir_count):
        pose = pd.read_csv(DataSet.TRANS_DATA_PATH + "/" + dir_count + "/" + str(keypoint_a) + '.csv', index_col=0)
        joi_a = pose.loc[pose.index % self.SKIP == 0, 'x':'y']
        A = joi_a.values

        pose = pd.read_csv(DataSet.TRANS_DATA_PATH + "/" + dir_count + "/" + str(keypoint_b) + '.csv', index_col=0)
        joi_b = pose.loc[pose.index % self.SKIP == 0, 'x':'y']
        B = joi_b.values

        return A, B

    # 角速度の計算結果を出力する
    def calculate_angular_velocity(self, key_point_a, key_point_b, dir_count):
        A, B = self.select_key_points(key_point_a, key_point_b, dir_count)

        angular_velocity = []
        for i in range(1, len(A)):
            angular_velocity.append(
                Calculation.AngularVelocity(self, A[i - 1], A[i], B[i - 1], B[i], self.SKIP))

        return angular_velocity

    # 未使用
    def calculate_angle(self, key_point_a, key_point_b, dir_count):
        A, B = self.select_key_points(key_point_a, key_point_b, dir_count)

        angle = []
        for i in range(1, len(A)):
            trans_keypoint = Translation(A[i - 1], A[i], B[i])
            angle.append(
                Angle(A[i - 1], B[i - 1], trans_keypoint))

        return angle

    # 未使用
    def calculate_speed(self, key_point, dir_count):
        A = self.one_key_point(key_point, dir_count)
        speed = []
        for i in range(1, len(A)):
            dist = Travel(A[i - 1], A[i])
            speed.append(Calculation.Speed(self, dist, self.SKIP))
        return speed

    # 未使用
    def calculate_travel(self, key_point, dir_count):
        A = self.one_key_point(key_point, dir_count)
        travel = []
        for i in range(1, len(A)):
            travel.append(Travel(A[i - 1], A[i]))
        return travel

    # 角速度のデータセットをcsvファイルに出力する
    def out_csv_file(self):
        count = file_count(self.KINEMATIC_DATA_PATH)
        data_set = []
        for i in range(count):
            datas = []
            path_list = glob.glob(self.KINEMATIC_DATA_PATH + '/' + str(i).zfill(3) + '/*')
            for j in path_list:
                kinematic_csv = pd.read_csv(j, usecols=[1])
                d = kinematic_csv.values
                for l in d:
                    datas.append(l[0])
            data_set.append(datas)
        kine = pd.DataFrame(data_set)
        kine.to_csv("./kinematic_dataset.csv")


KinematicData()
