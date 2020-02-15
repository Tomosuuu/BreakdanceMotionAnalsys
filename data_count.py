import os
import matplotlib.pyplot as plt


def file_count(path):
    count = os.listdir(path)
    return len(count)


# print(file_count("./Original_coordinate"))

ori = "./Original_coordinate"

num = []
video = []

for i in range(file_count(ori)):
    count = 0
    video.append(i+1)
    with open(ori + "/" + str(i).zfill(3) + "/Heart.csv") as f:
        for line in f:
            count += 1
    num.append(count - 1)


print(num)
m = num.index(min(num))
plt.rcParams["font.size"] = 30
fig, ax = plt.subplots(figsize=(20, 10), dpi=200)
bar_list = ax.bar(video, num)
bar_list[m].set_color("red")
plt.xlabel("Number of video data")
plt.ylabel("Number of data")
plt.savefig("./Graph/data_count.png")


