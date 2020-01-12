from KinematicDataCalculation import Calculation, Translation, Angle, Travel
from MakeDataSet import DataSet, file_count
import pandas as pd
import itertools
import os


class KinematicData(Calculation):
    skip = 10
    Keypoints = ['Nose',
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
    keypoints_perm = list(itertools.permutations(Keypoints, 2))
    Center_Point = Keypoints[1]
    Kinematic_data_path = "./Kinematic_Data"

    def __init__(self):
        kumiawase = self.Combination(self.Center_Point)
        os.makedirs(self.Kinematic_data_path, exist_ok=True)
        count = file_count(DataSet.Trans_data_path)
        for i in range(count):
            data_path = self.Kinematic_data_path + "/" + str(i).zfill(3)
            os.makedirs(data_path, exist_ok=True)
            dir_count = str(i).zfill(3)
            for a, b in kumiawase:
                kinematic_data = pd.DataFrame()
                kinematic_data['Angular_Velocity'] = self.calculate_angularvelocity(a, b, dir_count)
                kinematic_data['Angle'] = self.calculate_angle(a, b, dir_count)
                kinematic_data.to_csv(data_path + "/" + a + " - " + b + ".csv")

    def Combination(self, center):
        Combination = []
        for i in self.Keypoints:
            if i != center:
                Combination.append([center,i])
        return Combination


    def one_keypoint(self, keypoint_a, dir_count):
        pose = pd.read_csv(DataSet.Trans_data_path + "/" + dir_count + "/" + str(keypoint_a) + '.csv', index_col=0)
        joi_a = pose.loc[pose.index % self.skip == 0, 'x':'y']
        A = joi_a.values

        return A

    def select_keypoint(self, keypoint_a, keypoint_b, dir_count):
        pose = pd.read_csv(DataSet.Trans_data_path + "/" + dir_count + "/" + str(keypoint_a) + '.csv', index_col=0)
        joi_a = pose.loc[pose.index % self.skip == 0, 'x':'y']
        A = joi_a.values

        pose = pd.read_csv(DataSet.Trans_data_path + "/" + dir_count + "/" + str(keypoint_b) + '.csv', index_col=0)
        joi_b = pose.loc[pose.index % self.skip == 0, 'x':'y']
        B = joi_b.values

        return A, B

    def calculate_angularvelocity(self, keypoint_a, keypoint_b, dir_count):
        A, B = self.select_keypoint(keypoint_a, keypoint_b, dir_count)

        angular_velocity = []
        for i in range(1, len(A)):
            angular_velocity.append(
                Calculation.AngularVelocity(self, A[i - 1], A[i], B[i - 1], B[i], self.skip))

        return angular_velocity

    def calculate_angle(self, keypoint_a, keypoint_b, dir_count):
        A, B = self.select_keypoint(keypoint_a, keypoint_b, dir_count)

        angle = []
        for i in range(1, len(A)):
            trans_keypoint = Translation(A[i - 1], A[i], B[i])
            angle.append(
                Angle(A[i - 1], B[i - 1], trans_keypoint))

        return angle

    def calculate_speed(self, keypoint, dir_count):
        A = self.one_keypoint(keypoint, dir_count)
        speed = []
        for i in range(1, len(A)):
            dist = Travel(A[i - 1], A[i])
            speed.append(Calculation.Speed(self, dist, self.skip))
        return speed

    def calculate_travel(self, keypoint, dir_count):
        A = self.one_keypoint(keypoint, dir_count)
        travel = []
        for i in range(1, len(A)):
            travel.append(Travel(A[i - 1], A[i]))
        return travel


KinematicData()
