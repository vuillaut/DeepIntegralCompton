import tensorflow as tf
from tensorflow import keras
import numpy as np

def defineNetwork():
  model = keras.models.Sequential()
  model.add(keras.layers.InputLayer(input_shape=[None, 3]))
  for rate in (1, 2, 4, 8) * 2:
    model.add(keras.layers.Conv1D(filters=20, kernel_size=2, padding="causal", activation="relu", dilation_rate=rate))
  model.add(keras.layers.Conv1D(filters=10, kernel_size=1))
  model.add(keras.layers.Conv1D(filters=2, kernel_size=1))
  return model

def runModelAndReturnPrediction(targetPhi,targetTheta,anglesTrain,anglesTest, epochs=20, sizeofBatch=120):
  Y_train = np.transpose([targetPhi,targetTheta])
  X_train = anglesTrain
  
  #build tensor versions for target to fit the dimensionality requirements from the WaveNet ouputs
  TensYtrain = Y_train[:,np.newaxis,:]
  model = defineNetwork()
  model.compile(loss="mse", optimizer="adam", metrics=["mse"])
  model.fit(X_train, TensYtrain, epochs=epochs, batch_size=sizeofBatch)
  Y_pred_train = model.predict(X_train)
  Y_pred_test = model.predict(anglesTest)
  return Y_pred_train, Y_pred_test, model
