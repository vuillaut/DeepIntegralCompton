import os
from pathlib import Path
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import deepcompton
from deepcompton.utils import angular_separation


import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization, Dense, Flatten
from sklearn.model_selection import train_test_split

import numpy as np
import pickle as pkl

def angular_loss(y_true, y_pred):
    return -1. * (tf.math.sin(y_true[:,0])*tf.math.sin(y_pred[:,0])*
                  tf.math.cos(y_true[:,1]-y_pred[:,1])+
                  tf.math.cos(y_pred[:,0])*tf.math.cos(y_true[:,0]))
def angle(yt,yp):
    return tf.math.acos(-1.*angular_loss(yt,yp)) * 180. / np.pi

from sklearn.preprocessing import scale
def standardize(x):
    flat_x=[]
    for i in range(x.shape[0]):
        flat_x.append(x[i].flatten())
    flat_x=np.array(flat_x)

    flat_x = scale(flat_x)

    new_x = [flat_x[i].reshape(180,45,1) for i in range(flat_x.shape[0])]
    new_x = np.array(new_x)
    return x

class BaseModel1:
    def __init__(self, name="model", lr=1.e-4, max_epochs=1000, patience = 1):
        self.name = name
        self.lr = lr
        self.max_epochs = max_epochs
        self.patience = patience
    def get_model(self):
        model = tf.keras.Sequential()
        model.add(Conv2D(32,3,input_shape=(180,45,1), activation="relu"))
        model.add(BatchNormalization())
        model.add(Conv2D(64,3, activation="relu"))
        model.add(BatchNormalization())
        model.add(Conv2D(128,3, activation="relu"))
        model.add(BatchNormalization())
        model.add(Conv2D(256,3, activation="relu"))
        model.add(BatchNormalization())
 
        model.add(Flatten())
        model.add(Dense(512,activation="relu"))
        model.add(BatchNormalization())
        model.add(Dense(256,activation="relu"))
        model.add(BatchNormalization())
        model.add(Dense(128,activation="relu"))
        model.add(BatchNormalization())
        model.add(Dense(64,activation="relu"))
        model.add(BatchNormalization())
  
        model.add(Dense(2, activation="relu"))
        return model
    def train(self, x_train, y_train, x_test, y_test):
        if not os.path.exists("./models/{}".format(self.name)):
            os.system("mkdir -p ./models/{}".format(self.name))
 
        callbacks =[
                tf.keras.callbacks.ModelCheckpoint("./models/{}/weights.hdf5".format(self.name), monitor="val_loss"),
                tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=self.patience),
        ]
        
        model = self.get_model()

        model.compile(
                optimizer = tf.keras.optimizers.Adam(self.lr),
                loss=angular_loss,
                metrics=[angular_loss,"mean_squared_error",angle],
        )

        hist = model.fit(x_train, y_train, batch_size=256, epochs=self.max_epochs, callbacks=callbacks, validation_split=.2)
       
        # save the history
        pkl.dump(hist.history, open("./models/{}/hist.pkl".format(self.name), "wb"))
        self.make_test_outputs(model, x_test, y_test, hist)
    def make_test_outputs(self, model, x_test, y_test, history):
        y_pred = model(x_test).numpy()
        angular_seps=angular_separation(y_test[:,0],y_test[:,1],y_pred[:,0],y_pred[:,1]) * 180. / np.pi
        angular_seps = np.array(angular_seps)
        plt.figure()
        plt.hist(angular_seps, bins=100)
        print("Mean angular separation : {}".format(np.mean(angular_seps)))
        plt.title("Angular separation after training {}".format(self.name))
        plt.xlabel("Angular separation (deg)")
        plt.savefig("./models/{}/angular_separation_distribution.png".format(self.name))

        plt.figure()
        plt.plot(history.history["loss"], label="loss")
        plt.plot(history.history["val_loss"], label="val_loss")
        plt.title("Training loss {}".format(self.name))
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.savefig("./models/{}/loss.png".format(self.name))

        plt.figure()
        plt.plot(history.history["angle"], label="separation")
        plt.plot(history.history["val_angle"], label="val_separation")
        plt.title("Angular separation {}".format(self.name))
        plt.xlabel("Epochs")
        plt.ylabel("Angular separation (deg)")
        plt.savefig("./models/{}/angular_separation.png".format(self.name))



        

#from deepcompton.datasets.single_source_densities import SingleSourceDensityDataset
if __name__=="__main__":
    import sys
    name = sys.argv[1]
    lr = float(sys.argv[2])
    maxep = int(sys.argv[3])
    patience = int(sys.argv[4])
    datapath=sys.argv[5]
    # load the data here
    datapath = "UncertaintiesDataset.pkl"
    x,y= pkl.load(open(datapath, "rb"))
    x = np.array(x).reshape(len(x),180,45,1)
    y = np.radians(np.array(y))[:,:2]

    # standardize the data
    x = standardize(x)
    
    # train and testing data
    x_train,x_test, y_train,y_test = train_test_split(x, y, shuffle=True)
    y_train = tf.convert_to_tensor(y_train)
    x_train = tf.convert_to_tensor(x_train)

    m = BaseModel1(name, lr, maxep, patience)
    m.train(x_train, y_train, x_test, y_test)
