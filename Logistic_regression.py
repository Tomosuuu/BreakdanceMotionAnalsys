import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import scipy.stats


d = pd.read_csv("kinematic_dataset.csv", index_col=0)
data = d.values

t = pd.read_csv("Evaluate_Flare/evaluate-evaluate.csv", index_col=0)
target = t.T.values[0]

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)

# X_train = scipy.stats.zscore(X_train)
# X_test = scipy.stats.zscore(X_test)

clf = LogisticRegression()
clf.fit(X_train, y_train)

print("x_train:",X_train)
print("x_test:",X_test)
print(y_train)

print("----")
print(y_test)
print(clf.predict(X_test))

print("----")
print("score:",clf.score(X_test, y_test))


xjiku = []
for i in range(len(clf.predict(X_test))):
    xjiku.append(i)

plt.figure(figsize=(20,10),dpi=200)
plt.title("Nikkei Stock Average")
plt.xlabel('day')
plt.ylabel('stock price')
plt.plot(xjiku,clf.predict(X_test),linewidth=4,color="orange",label='predict value')
plt.scatter(xjiku, y_test,label='test_data')
plt.plot(xjiku, y_test,label='test_data')
plt.legend()
plt.show()