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

    def __init__(self):
        joint_perm = list(itertools.permutations(self.Joint,2))
        for p in joint_perm:
            an = pd.DataFrame(self.select_and_calculate_angularvelocity(p[0], p[1]), columns=['Angular_Velocity'])
            an.to_csv("Kinematic_Data/000/"+p[0]+" - "+p[1]+".csv")

    def select_and_calculate_angularvelocity(self, joint_a, joint_b):
        pose = pd.read_csv('Joint_Coordinate/000/'+str(joint_a)+'.csv', index_col=0)
        joi_a = pose.loc[pose.index % self.skip == 0, 'x':'y']
        A = joi_a.values

        pose = pd.read_csv('Joint_Coordinate/000/'+str(joint_b)+'.csv', index_col=0)
        joi_b = pose.loc[pose.index % self.skip == 0, 'x':'y']
        B = joi_b.values

        AngularVelocity_list = []
        for i in range(1, len(A)):
            a = Calculation.AngularVelocity(self, A[i - 1], A[i], B[i - 1], B[i], self.skip)
            AngularVelocity_list.append(a)

        return AngularVelocity_list


KinematicData()
