from _csv import reader
import csv
import numpy as np

def readlabel():
    # f = open("12piece_graph_labels.txt", encoding="utf-8")
    f = open("../balanceData_graph_labels.txt", encoding="utf-8")
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


def loadseqdata(file):
    '''数据预处理函数'''
    with open(file, 'r') as f:
        '''数据按行读取'''
        data = list(reader(f))
        # print(data)
        for i in range(len(data)):
            data[i] = np.array(data[i]).astype(dtype=int).tolist()
        # data = np.array(data).astype(dtype=int).tolist()
    '''转化为numpy数组'''
    data = np.array(data)

    print(data)
    return data

if __name__ == '__main__':
    file = '../clickseq.csv'
    # loadseqdata(file)
    readlabel()