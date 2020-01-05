import pandas as pd
import numpy as np
import os


class DataSet:
    Joint = {'Nose': [],
             'Heart': [],
             'right_shoulder': [],
             'right_elbow': [],
             'right_wrist': [],
             'left_shoulder': [],
             'left_elbow': [],
             'left_wrist': [],
             'right_hip': [],
             'right_knee': [],
             'right_ankle': [],
             'left_hip': [],
             'left_knee': [],
             'left_ankle': [],
             'right_eye': [],
             'left_eye': [],
             'right_ear': [],
             'left_ear': []}

    def __init__(self):
        pose = self.make_dataset()
        self.sepalate_data(pose)
        self.linear_transformation()

    def make_dataset(self):
        # path指定
        path = "./pose_csv"
        files = os.listdir(path)
        count = len(files)
        # print(count)

        # csvデータを統括して読み込み
        for i in range(count):
            num = str(i).zfill(3)
            pose = pd.read_csv('pose_csv/' + num + '.csv', header=None)
        # print(pose)
        return pose

    # keypointごとに振り分ける
    def sepalate_data(self, pose):
        pose_datas = pose.values
        for i in range(len(pose_datas)):
            if i % 18 == 0:
                self.Joint['Nose'].append(pose_datas[i])
            elif i % 18 == 1:
                self.Joint['Heart'].append(pose_datas[i])
            elif i % 18 == 2:
                self.Joint['right_shoulder'].append(pose_datas[i])
            elif i % 18 == 3:
                self.Joint['right_elbow'].append(pose_datas[i])
            elif i % 18 == 4:
                self.Joint['right_wrist'].append(pose_datas[i])
            elif i % 18 == 5:
                self.Joint['left_shoulder'].append(pose_datas[i])
            elif i % 18 == 6:
                self.Joint['left_elbow'].append(pose_datas[i])
            elif i % 18 == 7:
                self.Joint['left_wrist'].append(pose_datas[i])
            elif i % 18 == 8:
                self.Joint['right_hip'].append(pose_datas[i])
            elif i % 18 == 9:
                self.Joint['right_knee'].append(pose_datas[i])
            elif i % 18 == 10:
                self.Joint['right_ankle'].append(pose_datas[i])
            elif i % 18 == 11:
                self.Joint['left_hip'].append(pose_datas[i])
            elif i % 18 == 12:
                self.Joint['left_knee'].append(pose_datas[i])
            elif i % 18 == 13:
                self.Joint['left_ankle'].append(pose_datas[i])
            elif i % 18 == 14:
                self.Joint['left_eye'].append(pose_datas[i])
            elif i % 18 == 15:
                self.Joint['right_eye'].append(pose_datas[i])
            elif i % 18 == 16:
                self.Joint['right_ear'].append(pose_datas[i])
            elif i % 18 == 17:
                self.Joint['left_ear'].append(pose_datas[i])

    def linear_transformation(self):
        feature = ['x','y','trust']
        for key in self.Joint.keys():
            joint = pd.DataFrame(self.Joint[key], columns=feature)
            joint['x'] = joint['x'].where(joint['trust'] != 0.0)
            joint['y'] = joint['y'].where(joint['trust'] != 0.0)
            if joint.isnull().values.sum() != 0:
                joint_in = joint.interpolate()
                joint_in.to_csv("Joint_Coordinate/000/"+str(key)+".csv")
            else:
                joint.to_csv("Joint_Coordinate/000/" + str(key) + ".csv")


DataSet()
