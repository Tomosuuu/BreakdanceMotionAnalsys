import keras
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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
    keras.layers.Dense(216, activation='relu', input_shape=(204,)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(216, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(50, activation='relu'),
    # keras.layers.Dense(600, activation='relu'),
    # keras.layers.Dense(300, activation='relu'),
    keras.layers.Dense(6, activation='softmax')
])

model.summary()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train,
                    epochs=300,
                    batch_size=32,
                    validation_data=(X_test, y_test))

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)

print('\nTest accuracy:', test_acc)

expect = model.predict(X_test)
print(y_test)
# print(expect)
print(np.argmax(expect[0]),np.argmax(expect[1]),np.argmax(expect[2]),np.argmax(expect[3]),np.argmax(expect[4]),np.argmax(expect[5]),np.argmax(expect[6]))

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()