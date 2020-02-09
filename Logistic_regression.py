import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from MakeKinematicData import KinematicData
import os
from sklearn.metrics import mean_squared_error


def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)


class logistic():
    def __init__(self):
        KinematicData()
        os.makedirs("./Graph", exist_ok=True)
        s = []
        r = []
        count = 10
        score, rmse, ke = self.logisticreg()
        key = []
        j = 0
        for i in range(len(ke[0])):
            key.append(ke[0][i])
            if i % 24 == 0 and i != 0:
                plt.figure(figsize=(20, 10), dpi=200)
                plt.bar(range(len(key)),key)
                plt.savefig("./Graph/" + str(j) + ".png")
                j += 1
                key = []
        # for i in range(count):
        #     score, rmse, ke = self.logisticreg()
        #     s.append(score)
        #     r.append(rmse)
        #     print(len(ke[0]))
        #     plt.figure(figsize=(20, 10), dpi=200)
        #     plt.bar(range(len(ke[0])),ke[0])
        #     plt.savefig("./Graph/" + str(i) + ".png")
        # plt.figure(figsize=(20, 10), dpi=200)
        # plt.xlabel('count', fontsize=28)
        # plt.ylabel('score', fontsize=28)
        # plt.ylim(0.0,1.0)
        # plt.legend()
        # plt.plot(range(1,count+1), s, linewidth=4, color="orange")
        # plt.savefig("./Graph/" + "score" + ".png")
        #
        # plt.figure(figsize=(20, 10), dpi=200)
        # plt.xlabel('count', fontsize=28)
        # plt.ylabel('rmse', fontsize=28)
        # plt.legend()
        # plt.plot(range(1,count+1), r, linewidth=4, color="black")
        # plt.savefig("./Graph/" + "rmse" + ".png")

        # print("MAX:{}".format(max(s)))
        # print("MIN:{}".format(min(s)))
        # print("MEAN.{}".format(sum(s)/len(s)))
        # print("MAX:{}".format(max(r)))
        # print("MIN:{}".format(min(r)))
        # print("MEAN.{}".format(sum(r) / len(r)))

    def logisticreg(self):
        d = pd.read_csv("kinematic_dataset.csv", index_col=0)
        data = d.values

        t = pd.read_csv("Evaluate_Flare/evaluate-evaluate.csv", index_col=0)
        target = t.T.values[0]

        X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=3)

        stdsc = StandardScaler()
        X_train = stdsc.fit_transform(X_train)
        X_test = stdsc.transform(X_test)
        print(X_train)

        clf = LogisticRegression()
        clf.fit(X_train, y_train)

        # print("x_train:",X_train)
        # print("x_test:",X_test)
        # print(y_train)
        #
        # print("----")
        # print(y_test)
        # print(clf.predict(X_test))
        #
        # print("----")
        # print("score:",clf.score(X_test, y_test))
        regression_coefficient = clf.coef_
        segment = clf.intercept_
        print("回帰係数:{}".format(regression_coefficient))
        print("切片:{}".format(segment))
        print("決定係数:{}".format(clf.score(X_test, y_test)))
        # return clf.score(X_test, y_test)
        # return clf.score(X_test, y_test), mean_squared_error(clf.predict(X_test), y_test)
        rmse = np.sqrt(mean_squared_error(clf.predict(X_test), y_test))
        print("rmse:{}".format(rmse))

        return clf.score(X_test,y_test), rmse, regression_coefficient


logistic()

