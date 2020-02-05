import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import cross_val_score
import scipy.stats
import pandas as pd


d = pd.read_csv("kinematic_dataset.csv", index_col=0)
data = d.values

t = pd.read_csv("Evaluate_Flare/evaluate-evaluate.csv", index_col=0)
target = t.T.values[0]


x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.1)

# x_train = scipy.stats.zscore(x_train)
# x_test = scipy.stats.zscore(x_test)

params_cnt = 20
params = {"C":np.logspace(0,1,params_cnt), "epsilon":np.logspace(-1,1,params_cnt)}

gridsearch = GridSearchCV(svm.SVR(kernel="rbf"), params, scoring="r2", return_train_score=True).fit(x_train, y_train)

# reg = svm.SVR(kernel='rbf', C=1e3, gamma=0.00000001).fit(x_train, y_train)
rbf_reg = gridsearch.predict(x_test)



print("x_train:",x_train)
print("x_test:",x_test)
print(y_train)

print("----")
print(y_test)
print(rbf_reg)

print("C, εのチューニング")
print("最適なパラメーター =", gridsearch.best_params_)
print("精度 =", gridsearch.best_score_)
print()


xjiku = []
for i in range(len(rbf_reg)):
    xjiku.append(i)
#
plt.figure(figsize=(20,10),dpi=200)
plt.title("Nikkei Stock Average")
plt.xlabel('day')
plt.ylabel('stock price')
plt.plot(xjiku,rbf_reg,linewidth=4,color="orange",label='predict value')
plt.scatter(xjiku, y_test,label='test_data')
plt.plot(xjiku, y_test,label='test_data')
plt.legend()
plt.show()

# score = cross_val_score()


