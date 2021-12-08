import tensorflow as tf
from tensorflow import keras
import numpy as np

def defineNetwork():
  model = keras.models.Sequential([
    keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 3]),
    keras.layers.Masking(mask_value=-0.99),
    keras.layers.BatchNormalization(),
    keras.layers.SimpleRNN(20, return_sequences=False),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(2)
])
  return model

def runModelAndReturnPrediction(targetPhi,targetTheta,anglesTrain,anglesTest, epochs=20, sizeofBatch=120):
  Y_train = np.transpose([targetPhi,targetTheta])
  X_train = anglesTrain
  model = defineNetwork()
  model.compile(loss="mse", optimizer="adam", metrics=["mse"])
  model.fit(X_train, Y_train, epochs=epochs, batch_size=sizeofBatch)
  Y_pred_train = model.predict(X_train)
  Y_pred_test = model.predict(anglesTest)
  return Y_pred_train, Y_pred_test, model
