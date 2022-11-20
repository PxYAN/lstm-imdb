
# LSTM for sequence classification in the IMDB dataset
import random
from copy import deepcopy
from random import shuffle

import numpy as np
import pymysql
import tensorflow as tf
from keras import Model
from keras.callbacks import EarlyStopping
from keras.layers import Dropout, Bidirectional
from keras.optimizers import Adam, Adagrad, RMSprop, Adadelta, SGD
from matplotlib import pyplot as plt
from numpy import shape
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing import sequence
# fix random seed for reproducibility
from load import loadseqdata, readlabel
from imblearn.under_sampling import RandomUnderSampler


tf.random.set_seed(7)
# load the dataset but only keep the top n words, zero the rest
#参数 num_words=5000 的意思是仅保留训练数据中前 5000个最常出现的单词。低频单词将被舍弃
# top_words = 5000
top_words = 7
# (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)


# data = loadseqdata('clickseq.csv')
data = loadseqdata('../clickseq-balance.csv')
# copydata = data
# data = np.append(data,copydata)
labeldata = readlabel()
# copylabeldata = labeldata
# labeldata = np.append(labeldata,copylabeldata)

#打乱数据
# train_row = list(range(len(labeldata)))
# random.shuffle(train_row)
# data = data[train_row]
# labeldata = labeldata[train_row]

# X_train = data[0:2250]
# X_test = data[2250:]
# labeldata = readlabel()
# y_train = labeldata[0:2250]
# y_test = labeldata[2250:]

X_train = data[0:741]
X_test = data[741:]
y_train = labeldata[0:741]
y_test = labeldata[741:]
X_all = data[0:1059]

# X_train = data[0:1482]
# X_test = data[1482:]
# y_train = labeldata[0:1482]
# y_test = labeldata[1482:]

# data,labeldata = shuffle(data,labeldata)
# 欠采样
# rus = RandomUnderSampler(random_state=0)
# X_resampled, y_resampled = rus.fit_resample(X_train.reshape(-1, 1), y_train.reshape(-1, 1))
# Counter(y_resampled)
# print(y_resampled)
# print(X_resampled)
# print(shape(X_resampled))

# 查看样本数量
print("Training entries: {}, labels: {}".format(len(X_train),len(y_train)))
# 影评文本已转换为整数，其中每个整数都表示字典中的一个特定字词。第一条影评如下所示:
print("x-train.type",type(X_train))
print("y-train.type",type(y_train))
# print(X_train[0])
# print(len(X_train[0]))
# print(len(X_train[1]))
# 第一条影评的标签
# print(y_train[0])


# truncate and pad input sequences
#截断和填充输入序列，以便它们具有相同的长度以进行建模。
# 3138
# max_review_length = 500
max_review_length = 3138
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
X_all = sequence.pad_sequences(X_all, maxlen=max_review_length)

# create the model
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, mask_zero=True, input_length=max_review_length))
# model.add(LSTM(50))
# model.add(GRU(16)) # 128 16
model.add(
    Bidirectional(
        LSTM(
            units=20,
            return_sequences=True
        ),
        input_shape=(embedding_vecor_length, embedding_vecor_length)
    )
)
model.add(
    Bidirectional(
        LSTM(units=20)
    )
)
model.add(Dropout(0.0))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.0))
model.add(Dense(1,activation='sigmoid')) #sigmoid relu softmax

# optimizer = Adam(learning_rate=0.005,amsgrad=True) #5e-04
optimizer = Adam(lr=0.01) #5e-04

# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])   #mean_square_error

print(model.summary())
# 添加早停
callback = EarlyStopping(monitor="val_loss", patience=10, verbose=0, mode='min')

# h = model.fit(X_train, y_train, epochs=20, batch_size=128,validation_split=0.2) # 64
# h = model.fit(X_train, y_train, epochs=50, batch_size=256,validation_split=0.2,callbacks=[callback]) # 64 90 32
h = model.fit(X_train, y_train, epochs=120, batch_size=128)

history = h.history
epochs = range(len(history['accuracy']))


# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print(len(scores))
print("Accuracy: %.2f%%" % (scores[1]*100))


layer_model = Model(inputs=model.input, outputs=model.layers[2].output)
output = layer_model.predict(X_all)
print(output)
print(output.shape)

# xx = model.predict(X_all)
# print(model.predict(X_all))
# print(model.predict(X_all).shape)
np.savetxt(r'E:\python program\lstm-imdb\Bi-LSTM\bilstm-feature-combine.csv',output,delimiter=',', fmt=('%f'))

plt.plot(epochs, history['accuracy'], 'b', label='Train accuracy')
# plt.plot(epochs, history['val_accuracy'], 'r', label='Val accuracy')
plt.show()
plt.plot(epochs, history['loss'], 'b', label='Train loss')
# plt.plot(epochs, history['val_loss'], 'r', label='Val loss')
plt.show()

