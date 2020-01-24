import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
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

reg = svm.SVR(kernel='rbf', C=1e3, gamma=0.1).fit(x_train, y_train)
rbf_reg = reg.predict(x_test)

print("x_train:",x_train)
print("x_test:",x_test)
print(y_train)

print("----")
print(y_test)
print(rbf_reg)


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


