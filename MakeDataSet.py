import copy
import os
import pandas as pd


# directory内のファイルをカウントする
def file_count(path):
    count = os.listdir(path)
    return len(count)

# 動作がスタートしているフレームを開始点とする
def detect_start(dir_num):
    start = pd.read_csv("./Evaluate_Flare/start-start_frame.csv", usecols=["start"])
    start_frame = start.T.values[0]
    return start_frame[dir_num]

# 振り分ける
def separate_data(pose, COPIED_KEY_POINTS):
    pose_data = pose.values
    for i in range(len(pose_data)):
        if i % 18 == 0:
            COPIED_KEY_POINTS['Nose'].append(pose_data[i])
        elif i % 18 == 1:
            COPIED_KEY_POINTS['Heart'].append(pose_data[i])
        elif i % 18 == 2:
            COPIED_KEY_POINTS['right_shoulder'].append(pose_data[i])
        elif i % 18 == 3:
            COPIED_KEY_POINTS['right_elbow'].append(pose_data[i])
        elif i % 18 == 4:
            COPIED_KEY_POINTS['right_wrist'].append(pose_data[i])
        elif i % 18 == 5:
            COPIED_KEY_POINTS['left_shoulder'].append(pose_data[i])
        elif i % 18 == 6:
            COPIED_KEY_POINTS['left_elbow'].append(pose_data[i])
        elif i % 18 == 7:
            COPIED_KEY_POINTS['left_wrist'].append(pose_data[i])
        elif i % 18 == 8:
            COPIED_KEY_POINTS['right_hip'].append(pose_data[i])
        elif i % 18 == 9:
            COPIED_KEY_POINTS['right_knee'].append(pose_data[i])
        elif i % 18 == 10:
            COPIED_KEY_POINTS['right_ankle'].append(pose_data[i])
        elif i % 18 == 11:
            COPIED_KEY_POINTS['left_hip'].append(pose_data[i])
        elif i % 18 == 12:
            COPIED_KEY_POINTS['left_knee'].append(pose_data[i])
        elif i % 18 == 13:
            COPIED_KEY_POINTS['left_ankle'].append(pose_data[i])
        elif i % 18 == 14:
            COPIED_KEY_POINTS['left_eye'].append(pose_data[i])
        elif i % 18 == 15:
            COPIED_KEY_POINTS['right_eye'].append(pose_data[i])
        elif i % 18 == 16:
            COPIED_KEY_POINTS['right_ear'].append(pose_data[i])
        elif i % 18 == 17:
            COPIED_KEY_POINTS['left_ear'].append(pose_data[i])


class DataSet:
    KEY_POINTS = {'Nose': [],
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
    TRANS_DATA_PATH = "./Trans_Coordinate_Data"

    def __init__(self):
        self.make_dataset()

    # csvデータを統括して読み込み
    def make_dataset(self):
        os.makedirs(self.TRANS_DATA_PATH, exist_ok=True)
        count = file_count(self.PATH)

        for dir_num in range(count):
            dir_count = str(dir_num).zfill(3)
            path_in = self.PATH + '/' + dir_count
            start = detect_start(dir_num)
            end = file_count(path_in)
            COPIED_KEY_POINTS = copy.deepcopy(self.KEY_POINTS)
            pose = pd.DataFrame()
            for file_num in range(start, end):
                if os.path.isfile(path_in + '/' + str(file_num).zfill(3) + '.csv'):
                    POSE = pd.read_csv(path_in + '/' + str(file_num).zfill(3) + '.csv', header=None)
                    pose = pd.concat([pose, POSE])
            separate_data(pose, COPIED_KEY_POINTS)
            os.makedirs(self.TRANS_DATA_PATH + "/" + dir_count, exist_ok=True)
            self.linear_transformation(dir_count, COPIED_KEY_POINTS)

    # 線形変換する
    def linear_transformation(self, dir_count, COPIED_KEY_POINTS):
        feature = ['x', 'y', 'trust']
        for key in COPIED_KEY_POINTS.keys():
            key_points = pd.DataFrame(COPIED_KEY_POINTS[key], columns=feature)
            key_points['x'] = key_points['x'].where(key_points['trust'] != 0.0)
            key_points['y'] = key_points['y'].where(key_points['trust'] != 0.0)
            if key_points.isnull().values.sum() != 0:
                joint_in = key_points.interpolate()
                joint_in.to_csv(self.TRANS_DATA_PATH + "/" + dir_count + "/" + str(key) + ".csv")
            else:
                key_points.to_csv(self.TRANS_DATA_PATH + "/" + dir_count + "/" + str(key) + ".csv")


DataSet()
