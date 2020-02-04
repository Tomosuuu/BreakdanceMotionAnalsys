import keras
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

d = pd.read_csv("kinematic_dataset.csv", index_col=0)
data = d.values

t = pd.read_csv("Evaluate_Flare/evaluate-evaluate.csv", index_col=0)
target = t.T.values[0]

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.1)

# トレーニングデータの正規化
X_train_mean = X_train.mean(axis=0)
X_train_std = X_train.std(axis=0)
X_train -= X_train_mean
X_train /= X_train_std
#
# y_train = np.array(y_train, dtype=np.float)
# y_train_mean = y_train.mean()
# y_train_std = y_train.std()
# y_train -= y_train_mean
# y_train /= y_train_std
#
# # テストデータの正規化
# y_test = np.array(y_test, dtype=np.float)
X_test -= X_train_mean
X_test /= X_train_std
# y_test -= y_train_mean
# y_test /= y_train_std


model = keras.Sequential([
    keras.layers.Dense(425, input_dim=425, activation='relu'),
    # keras.layers.Dense(426, activation='relu'),
    # keras.layers.Dense(426, activation='relu'),
    # keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(6, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=200)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)

print('\nTest accuracy:', test_acc)

expect = model.predict(X_test)
print(y_test)
# print(expect)
print(np.argmax(expect[0]),np.argmax(expect[1]),np.argmax(expect[2]),np.argmax(expect[3]),np.argmax(expect[4]),np.argmax(expect[5]),np.argmax(expect[6]))
