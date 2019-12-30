import pandas as pd
import numpy as np
import os

JOINT = ['Nose', 'Heart', 'right shoulder', 'right elbow',
         'right wrist', 'left shoulder', 'left elbow', 'left wrist',
         'right hip', 'right knee', 'right ankle', 'left hip',
         'left knee', 'left ankle', 'right eye', 'left eye',
         'right ear', 'left ear']

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

# keypointごとに振り分ける
pose_datas = pose.values
joints = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
for i in range(len(pose_datas)):
    if i % 18 == 0:
        joints[0].append(list(pose_datas[i]))
    elif i % 18 == 1:
        joints[1].append(list(pose_datas[i]))
    elif i % 18 == 2:
        joints[2].append(list(pose_datas[i]))
    elif i % 18 == 3:
        joints[3].append(list(pose_datas[i]))
    elif i % 18 == 4:
        joints[4].append(list(pose_datas[i]))
    elif i % 18 == 5:
        joints[5].append(list(pose_datas[i]))
    elif i % 18 == 6:
        joints[6].append(list(pose_datas[i]))
    elif i % 18 == 7:
        joints[7].append(list(pose_datas[i]))
    elif i % 18 == 8:
        joints[8].append(list(pose_datas[i]))
    elif i % 18 == 9:
        joints[9].append(list(pose_datas[i]))
    elif i % 18 == 10:
        joints[10].append(list(pose_datas[i]))
    elif i % 18 == 11:
        joints[11].append(list(pose_datas[i]))
    elif i % 18 == 12:
        joints[12].append(list(pose_datas[i]))
    elif i % 18 == 13:
        joints[13].append(list(pose_datas[i]))
    elif i % 18 == 14:
        joints[14].append(list(pose_datas[i]))
    elif i % 18 == 15:
        joints[15].append(list(pose_datas[i]))
    elif i % 18 == 16:
        joints[16].append(list(pose_datas[i]))
    elif i % 18 == 17:
        joints[17].append(list(pose_datas[i]))
    elif i % 18 == 18:
        joints[18].append(list(pose_datas[i]))

jo0 = pd.DataFrame(joints[0])
jo1 = pd.DataFrame(joints[1])
jo2 = pd.DataFrame(joints[2])
jo3 = pd.DataFrame(joints[3])
jo4 = pd.DataFrame(joints[4])
jo5 = pd.DataFrame(joints[5])
jo6 = pd.DataFrame(joints[6])
jo7 = pd.DataFrame(joints[7])
jo8 = pd.DataFrame(joints[8])
jo9 = pd.DataFrame(joints[9])
jo10 = pd.DataFrame(joints[10])
jo11 = pd.DataFrame(joints[11])
jo12 = pd.DataFrame(joints[12])
jo13 = pd.DataFrame(joints[13])
jo14 = pd.DataFrame(joints[14])
jo15 = pd.DataFrame(joints[15])
jo16 = pd.DataFrame(joints[16])
jo17 = pd.DataFrame(joints[17])

datas = [jo0, jo1, jo2, jo3, jo4, jo5, jo6, jo7, jo8, jo9, jo10, jo11, jo12, jo13, jo14, jo15, jo16, jo17]

jointdata = pd.concat(
[jo0, jo1, jo2, jo3, jo4, jo5, jo6, jo7, jo8, jo9, jo10, jo11, jo12, jo13, jo14, jo15, jo16, jo17], axis=1)
jointdata.to_csv('joints.csv', header=False, index=False)

joi = pd.read_csv('joints.csv', header=None,
                  names=(
                      'x0', 'y0', 'h0',
                      'x1', 'y1', 'h1',
                      'x2', 'y2', 'h2',
                      'x3', 'y3', 'h3',
                      'x4', 'y4', 'h4',
                      'x5', 'y5', 'h5',
                      'x6', 'y6', 'h6',
                      'x7', 'y7', 'h7',
                      'x8', 'y8', 'h8',
                      'x9', 'y9', 'h9',
                      'x10', 'y10', 'h10',
                      'x11', 'y11', 'h11',
                      'x12', 'y12', 'h12',
                      'x13', 'y13', 'h13',
                      'x14', 'y14', 'h14',
                      'x15', 'y15', 'h15',
                      'x16', 'y16', 'h16',
                      'x17', 'y17', 'h17',))



joi['x17'] = joi['x17'].where(joi['h17'] != 0.0)
joi['y17'] = joi['y17'].where(joi['h17'] != 0.0)


print(joi.interpolate())