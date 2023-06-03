from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import math as m

def gen_data(n):
    nums = [i for i in range(2, n)]
    labels = [1 for i in range(2, n)]
    for i, x in enumerate(nums):
        for j in range(2, x):
            if x % j == 0:
                labels[i] = 0
                break
    return [np.array(nums), to_categorical(labels)]

def test_train_splits(x, y, test_size):
    split = int(len(x) * test_size)
    x_train, x_test = x[:split], x[split:]
    y_train, y_test = y[:split], y[split:]
    return x_train, x_test, y_train, y_test

data, labels = gen_data(1000)
x_train, x_test, y_train, y_test = test_train_splits(data, labels, 0.8)

model = Sequential()
model.add(Dense(32, input_dim=1, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=20, batch_size=32)

result = model.predict([6])
print(to_categorical(result, dtype="uint8"))
