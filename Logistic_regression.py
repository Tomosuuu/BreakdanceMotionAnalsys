import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

cancer = datasets.load_breast_cancer()

df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
df["label"] = cancer.target
df.head()

set(cancer.target)

X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, test_size=0.2, random_state=1234)

clf = LogisticRegression()
clf.fit(X_train, y_train)

print(clf.predict(X_test))

print(clf.score(X_train, y_train))

print(clf.score(X_test, y_test))
