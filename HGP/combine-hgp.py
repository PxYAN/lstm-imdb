import numpy as np
import pandas as pd

data1 = pd.read_csv('E:/python program/lstm-imdb/MLP/balance/test-balance-1.csv')
# data2 = pd.read_csv('E:/python program/lstm-imdb/GRU/gru-feature-output-pxy.csv')
data2 = pd.read_csv('E:\python program\lstm-imdb\HGP\hgp-feature.csv')
data2 = pd.DataFrame(data2)
# data2 = data2.iloc[:,-1]
# data1 = pd.DataFrame(data1)
print(data2)
print(data1)

xx2 = pd.merge(data1,data2,how="inner",left_index=True,right_index=True)
print(xx2)
# np.savetxt(r'E:\python program\lstm-imdb\LSTM\lstm-feature-combine-lstm-all.csv',xx2,delimiter=',', fmt=('%d,%f,%d,%f,%f,%d,%f,%d,%f'))
# np.savetxt(r'E:\python program\lstm-imdb\LSTM\lstm-feature-combine-lstm-all.csv',xx2,delimiter=',', fmt=('%d,%f,%d,%f,%f,%d,%f,%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f'))
# np.savetxt(r'E:\python program\lstm-imdb\LSTM\lstm-feature-combine-lstm-1-2.csv',xx2,delimiter=',', fmt=('%d,%f,%d,%f,%f,%d,%f,%d,%f'))
np.savetxt(r'E:\python program\lstm-imdb\HGP\hgp-feature-combine-hgp.csv',xx2,delimiter=',', fmt=('%d,%f,%d,%f,%f,%d,%f,%d,%f,%f'))
