try:
  import pymysql
except ImportError:
  print("Must install pymysql")
  !pip install pymysql
  import pymysql

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import gc

from sklearn.preprocessing import MinMaxScaler


# from keras.models import Sequential
# from keras.layers import Dense
# from keras.callbacks import EarlyStopping, ModelCheckpoint
import model.build_model as build_model
from sklearn.model_selection import train_test_split

def make_dataset(data, label, window_size=20):
  feature_list = []
  label_list = []
  
  for i in range(len(data) - window_size):
    feature_list.append(np.array(data.iloc[i:i+window_size]))
    label_list.append(np.array(label.iloc[i+window_size]))
  return np.array(feature_list), np.array(label_list)

# def build_model():
#   model = Sequential()
#   model.add(LSTM(16,
#                input_shape=(train_feature.shape[1], train_feature.shape[2]),
#                activation="relu",
#                return_sequences=False)
#          )
#   model.add(Dense(1))
#   return model

def _train(model, train_set, test_set, feature, label):
  train_feature = train_set[feature]
  train_label = train_set[label]

  test_feature = test_set[feature_cols]
  test_label = test_set[label_cols]

  train_feature, train_label = make_dataset(train_feature, train_label, 20)
  test_feature, test_label = make_dataset(test_feature, test_label, 20)
  x_train, x_valid, y_train, y_valid = train_test_split(train_feature, train_label, test_size=0.2)

  model.compile(loss='mean_squared_error', optimizer='adam')
  early_stop = EarlyStopping(monitor='val_loss', patience=5)

  history = model.fit(x_train, y_train,
                      epochs=200,
                      batch_size=6,
                      validation_data=(x_valid, y_valid),
                      callbacks=[early_stop],
                      verbose=True)

  pred = model.predict(test_feature)
  return pred

# 데이터베이스 접근
wti_hloc = None # /open your own database

scaler = MinMaxScaler()
scale_cols = df.columns[1:]

df_scaled = sclaer.fit_transform(wti_hloc[scale_cols])
df_scaled = pd.DataFrame(df_scaled, columns=scale_cols)


TEST_SIZE = 200
train = df_scaled[:-TEST_SIZE]
test = df_scaled[-TEST_SIZE:]

# Model 생성
model = build_model()

# Test1 --> 시가, 고가, 저가를 이용하여 종가 예측하기
feature_cols = ['price_open','price_high','price_low']  # 각 database의 시가, 고가, 저가를 Feature로
label_cols = ['price_close']                            # 각 database의 종가를 Label로

model1 = build_model()
pred1 = _train(model1, train, test, feature_cols, label_cols)

# Test 2
feature_cols = ['price_open','price_high','price_low','vol_k']  # 각 database의 시가, 고가, 저가, 총 거래량를 Feature로
label_cols = ['price_close']                                    # 각 database의 종가를 Label로

model2 = build_model()
pred2 = _train(model2, train, test, feature_cols, label_cols)

# Test 3
feature_cols = ['price_open', 'price_high', 'price_low','vol_k','diff_percent'] # 각 database의 시가, 고가, 저가, 총 거래량, 거래 증감률를 Feature로
label_cols = ['price_close']                                                    # 각 database의 종가를 Label로

model3 = build_model()
pred3 = _train(model3, train, test, feature_cols, label_cols)

# 결과 확인하기
plt.figure(figsize=(32, 18))
plt.plot(test[label_cols], '--', label='actual')
plt.plot(pred1, label='case1')
plt.plot(pred2, label='case2')
plt.plot(pred3, label='case3')
plt.legend()
plt.show()
         

