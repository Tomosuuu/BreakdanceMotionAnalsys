import keras
from keras.datasets import boston_housing

(X_train, y_train), (X_test, y_test) = boston_housing.load_data()

# トレーニングデータの正規化
X_train_mean = X_train.mean(axis=0)
X_train_std = X_train.std(axis=0)
X_train -= X_train_mean
X_train /= X_train_std

y_train_mean = y_train.mean()
y_train_std = y_train.std()
y_train -= y_train_mean
y_train /= y_train_std

# テストデータの正規化
X_test -= X_train_mean
X_test /= X_train_std
y_test -= y_train_mean
y_test /= y_train_std

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam',
              loss='mse',
              metrics=['mae'])

history = model.fit(X_train, y_train,
                    batch_size=1,
                    epochs=100,
                    verbose=1,
                    validation_data=(X_test, y_test))

expect = model.predict(X_test, batch_size=5)
print(expect)