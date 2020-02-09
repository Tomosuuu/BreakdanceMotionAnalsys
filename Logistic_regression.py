import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from MakeKinematicData import KinematicData
import os
from sklearn.metrics import mean_squared_error

plt.rcParams["font.size"] = 30

class Logistic:
    def __init__(self):
        KinematicData()
        os.makedirs("./Graph", exist_ok=True)
        score_list = []
        rmse_list = []
        count = 100
        for i in range(count):
            score, rmse = self.logisticreg()
            score_list.append(score)
            rmse_list.append(rmse)

        # plt.figure(figsize=(20, 10), dpi=200)
        # plt.xlabel('count')
        # plt.ylabel('score')
        # plt.ylim(0.0,1.0)
        # plt.legend()
        # plt.plot(range(1,count+1), score_list, linewidth=4, color="orange")
        # plt.savefig("./Graph/" + "score" + ".png")

        plt.figure(figsize=(20, 10), dpi=200)
        plt.ylabel('count')
        plt.xlabel('RMSE')
        plt.legend()
        plt.hist(rmse_list)
        plt.savefig("./Graph/" + "rmse" + ".png")
        print(rmse_list)
        print(score_list)

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
        # print(X_train)

        clf = LogisticRegression()
        clf.fit(X_train, y_train)

        # print("x_train:",X_train)
        # print("x_test:",X_test)
        # print(y_train)
        #
        # print("----")
        print(y_test)
        print(clf.predict(X_test))
        #
        # print("----")
        # print("score:",clf.score(X_test, y_test))
        # regression_coefficient = clf.coef_
        # segment = clf.intercept_
        # print("回帰係数:{}".format(regression_coefficient))
        # print("切片:{}".format(segment))
        print("決定係数:{}".format(clf.score(X_test, y_test)))
        # return clf.score(X_test, y_test)
        # return clf.score(X_test, y_test), mean_squared_error(clf.predict(X_test), y_test)
        rmse = np.sqrt(mean_squared_error(list(clf.predict(X_test)), list(y_test)))
        print("rmse:{}".format(rmse))

        return clf.score(X_test,y_test), rmse


Logistic()

