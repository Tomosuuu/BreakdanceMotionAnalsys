from KinematicDataCalculation import Calculation
import pandas as pd
import itertools


class KinematicData(Calculation):
    skip = 10
    Joint = ['Nose',
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
    joint_perm = list(itertools.permutations(Joint, 2))

    def __init__(self):
        for a, b in self.joint_perm:
            kinematic_data = pd.DataFrame()
            kinematic_data['Angular_Velocity'] = self.calculate_angularvelocity(a, b)
            kinematic_data['Angle'] = self.calculate_angle(a, b)
            # kinematic_data['Speed'] = self.calculate_speed()
            # kinematic_data['Travel'] = self.calculate_travel()

            kinematic_data.to_csv("Kinematic_Data/two_joint/000/" + a + " - " + b + ".csv")

        for joint in self.Joint:
            one_joint_kinematic_data = pd.DataFrame()
            one_joint_kinematic_data['Speed'] = self.calculate_speed(joint)
            one_joint_kinematic_data['Travel'] = self.calculate_travel(joint)
            one_joint_kinematic_data.to_csv("Kinematic_Data/one_joint/000/" + joint +".csv")

    def one_joint(self, joint_a):
        pose = pd.read_csv('Joint_Coordinate/000/' + str(joint_a) + '.csv', index_col=0)
        joi_a = pose.loc[pose.index % self.skip == 0, 'x':'y']
        A = joi_a.values

        return A

    def select_joint(self, joint_a, joint_b):
        pose = pd.read_csv('Joint_Coordinate/000/' + str(joint_a) + '.csv', index_col=0)
        joi_a = pose.loc[pose.index % self.skip == 0, 'x':'y']
        A = joi_a.values

        pose = pd.read_csv('Joint_Coordinate/000/' + str(joint_b) + '.csv', index_col=0)
        joi_b = pose.loc[pose.index % self.skip == 0, 'x':'y']
        B = joi_b.values

        return A, B

    def calculate_angularvelocity(self, joint_a, joint_b):
        A, B = self.select_joint(joint_a, joint_b)

        angular_velocity = []
        for i in range(1, len(A)):
            angular_velocity.append(
                Calculation.AngularVelocity(self, A[i - 1], A[i], B[i - 1], B[i], self.skip))

        return angular_velocity

    def calculate_angle(self, joint_a, joint_b):
        A, B = self.select_joint(joint_a, joint_b)

        angle = []
        for i in range(1, len(A)):
            trans_joint = Calculation.Translation(self, A[i - 1], A[i], B[i])
            angle.append(
                Calculation.Angle(self, A[i - 1], B[i - 1], trans_joint))

        return angle

    def calculate_speed(self, joint):
        A = self.one_joint(joint)
        speed = []
        for i in range(1, len(A)):
            dist = Calculation.Travel(self, A[i - 1], A[i])
            speed.append(Calculation.Speed(self, dist, self.skip))
        return speed

    def calculate_travel(self, joint):
        A = self.one_joint(joint)
        travel = []
        for i in range(1, len(A)):
            travel.append(Calculation.Travel(self, A[i - 1], A[i]))
        return travel



KinematicData()
