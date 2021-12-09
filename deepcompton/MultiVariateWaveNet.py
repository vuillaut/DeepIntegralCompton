import tensorflow as tf
from tensorflow import keras
import numpy as np
import tensorflow_probability as tfp
tfpl = tfp.layers
tfkl = keras.layers
tfd = tfp.distributions

def defineNetwork():
  prior = tfd.MultivariateNormalDiag(loc=tf.zeros(2))

  model = keras.models.Sequential()
  model.add(keras.layers.InputLayer(input_shape=[None, 3]))
  for rate in (1, 2, 4, 8) * 2:
    model.add(keras.layers.Conv1D(filters=20, kernel_size=2, padding="causal", activation="relu", dilation_rate=rate))
  model.add(keras.layers.Conv1D(filters=10, kernel_size=1))
  model.add(keras.layers.GlobalAveragePooling1D())
  model.add(tfkl.Dense(tfpl.MultivariateNormalTriL.params_size(2)))
  model.add(tfpl.MultivariateNormalTriL(2,activity_regularizer=tfpl.KLDivergenceRegularizer(prior)))
  return model

def runModelAndReturnPrediction(targetPhi,targetTheta,anglesTrain,anglesTest, epochs=20, sizeofBatch=120):
  Y_train = np.transpose([targetPhi,targetTheta])
  X_train = anglesTrain
  model = defineNetwork()
  negloglik = lambda x, distrib: -distrib.log_prob(x)
  model.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-3),loss=negloglik)
  model.fit(X_train, Y_train, epochs=epochs, batch_size=sizeofBatch)
  return model
