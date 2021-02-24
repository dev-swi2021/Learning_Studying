from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint

def build_model():
  model = Sequential()
  model.add(LSTM(16,
               input_shape=(train_feature.shape[1], train_feature.shape[2]),
               activation="relu",
               return_sequences=False)
         )
  model.add(Dense(1))
  return model
