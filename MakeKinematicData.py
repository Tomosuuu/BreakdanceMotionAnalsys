from KinematicDataCalculation import Calculation
import pandas as pd

class KinematicData(Calculation):
    def __init__(self):
        pose = pd.read_csv('Joint_Coordinate/000/Heart.csv', index_col=0)
        Heart = pose.loc[pose.index % 10 == 0, 'x':'y']
        A = Heart.values

        pose = pd.read_csv('Joint_Coordinate/000/left_hip.csv', index_col=0)
        left_hip = pose.loc[pose.index % 10 == 0, 'x':'y']
        B = left_hip.values

        for i in range(1,len(A)):
            Calculation.AngularVelocity(self, A[i-1], A[i], B[i-1], B[i], 10)

KinematicData()