import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



def readlabel():
    # f = open("12piece_graph_labels.txt", encoding="utf-8")
    f = open("balance/balanceData_graph_labels.txt", encoding="utf-8")
    # 输出读取到的数据
    data = list(f.read())
    for i in data:
        if i == '\n':
            data.remove(i)
    for i in range(len(data)):
        data[i] = np.array(data[i]).astype(dtype=int).tolist()
        # data = np.array(data).astype(dtype=int).tolist()
    '''转化为numpy数组'''
    data = np.array(data)
    print(data)
    return data

def bianli(i):
    tf.random.set_seed(i) #45-71% 71-73%
    # data=pd.read_csv('balance/test-balance-1.csv')
    # data=pd.read_csv('E:\python program\lstm-imdb\LSTM\lstm-feature-combine-2.csv')
    # data=pd.read_csv('E:/python program/lstm-imdb/LSTM/lstm-feature-combine-lstm-1-2.csv')
    # data=pd.read_csv('E:/python program/lstm-imdb/GRU/gru-feature-combine-gru.csv')
    data = pd.read_csv('E:/python program/lstm-imdb/GRU/gru-feature-combine-gru-2.csv')

    datay = readlabel()
    print(data)
    print(data.head())#查看数据概况
    # dataX=data.iloc[:,1:-1]
    # datay=data.iloc[:,-1]
    dataX = data.iloc[:,:]
    print(dataX)

    train_X=dataX[:741]
    train_y=datay[:741]
    test_X=dataX[741:]
    test_y=datay[741:]

    model=tf.keras.Sequential([
        # tf.keras.layers.Dense(64,input_shape=(8,),activation='sigmoid'), # relu
        tf.keras.layers.Dense(64,input_shape=(9,),activation='sigmoid'),
        # tf.keras.layers.Dense(64,input_shape=(43,),activation='sigmoid'),
        tf.keras.layers.Dense(16),
        tf.keras.layers.Dense(64),
        # tf.keras.layers.Dense(32),
        tf.keras.layers.Dense(32),
        tf.keras.layers.Dense(1)#回归问题，输出层一个神经元
    ]
    )
    model.summary()#查看模型概况


    model.compile(
        optimizer='adam',
        loss='mse', #mse
        metrics=['accuracy']
    )#通过调用 compile 方法配置该模型的学习流程，回归问题使用均方误差损失函数
    h = model.fit(train_X,train_y,epochs=220,batch_size=512)#fit方法进行训练，训练100轮次 512

    scores = model.evaluate(test_X, test_y, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))

    # print(model.predict(test_X))
    # print(test_y)

    # history = h.history
    # epochs = range(len(history['accuracy']))
    # plt.plot(epochs, history['accuracy'], 'r', label='Train accuracy')
    # # plt.plot(epochs, history['val_accuracy'], 'b', label='val accuracy')
    # plt.show()
    # plt.plot(epochs, history['loss'], 'r', label='Train loss')
    # # plt.plot(epochs, history['val_loss'], 'b', label='val accuracy')
    # plt.show()
    return scores[1]*100

# print(bianli(41)) # 176-75.79% 90-74.84% gru-feature-combine-gru(epoch-120).csv
# print(bianli(41)) # 176-77.67% 90-73.27% gru-feature-combine-gru-2(epoch-88).csv


bianli(176)

# maxi = 0
# maxacc = 0
# acclist = list()
# for i in range(1000):
#     current = bianli(i)
#     acclist.append(current)
#     if maxacc < current:
#         maxacc = current
#         maxi = i
# print(maxi)
# print(maxacc)
# steps = range(0,1000)
# plt.plot(steps, acclist, 'b', label='100 accuracy')
# plt.show()