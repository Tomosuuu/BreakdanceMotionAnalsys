import pandas as pd
import os


def file_count(path):
    count = os.listdir(path)
    return len(count)


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
    PATH = "./coordinate_csv"
    Trans_data_path = "./Trans_Coordinate_Data"

    def __init__(self):
        os.makedirs(self.Trans_data_path, exist_ok=True)
        self.make_dataset()

    def make_dataset(self):
        count = file_count(self.PATH)

        # csvデータを統括して読み込み
        for dir_num in range(count):
            dir_count = str(dir_num).zfill(3)
            path_in = self.PATH + '/' + dir_count
            num = file_count(path_in)
            for file_num in range(num):
                if os.path.isfile(path_in + '/' + str(file_num).zfill(3) + '.csv'):
                    pose = pd.read_csv(path_in + '/' + str(file_num).zfill(3) + '.csv', header=None)
                self.separate_data(pose, dir_count)

    # keypointごとに振り分ける
    def separate_data(self, pose, dir_count):
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

        os.makedirs(self.Trans_data_path + "/" + dir_count, exist_ok=True)
        self.linear_transformation(dir_count)

    def linear_transformation(self, dir_count):
        feature = ['x', 'y', 'trust']
        for key in self.Joint.keys():
            joint = pd.DataFrame(self.Joint[key], columns=feature)
            joint['x'] = joint['x'].where(joint['trust'] != 0.0)
            joint['y'] = joint['y'].where(joint['trust'] != 0.0)
            if joint.isnull().values.sum() != 0:
                joint_in = joint.interpolate()
                joint_in.to_csv(self.Trans_data_path + "/" + dir_count + "/" + str(key) + ".csv")
            else:
                joint.to_csv(self.Trans_data_path + "/" + dir_count + "/" + str(key) + ".csv")

    def out_csv_file(self):
        kinematic_csv = "./Kinematic_data"
        dir_count = file_count(kinematic_csv)
        kinematic_dataset = pd.DataFrame()
        for i in range(dir_count):
            one_dataset = pd.DataFrame()
            dir_path = kinematic_csv + '/' + str(i).zfill(3)
            kine_count = file_count(dir_path)
            for j in range(kine_count):
                kinematic_data = pd.read_csv(dir_path + '/' + str(j).zfill(3) + '.csv')
                pd.concat(one_dataset,kinematic_data.T)
            pd.concat(kinematic_dataset,one_dataset)
        print(kinematic_dataset)


DataSet()
