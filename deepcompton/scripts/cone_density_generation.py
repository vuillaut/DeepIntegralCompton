import matplotlib.pyplot as plt
import numpy as np
import os
import pickle as pkl
print(os.getcwd())
from deepcompton.cones import make_cone_density

from threading import Lock

lock = Lock()

plt.rcParams.update({"font.size":14})
Ee = 511

z_isgri = 0
z_picsit = -8.68

n = 100

data_filename = "cone_density_data_full_rand_better.pkl"
if not os.path.exists(data_filename):
    # get cone density data for all files in dataset
    from multiprocessing import Pool, Manager
    manager = Manager()
    data=manager.list()    
    labels=manager.list()
    def get_data(f):
        for i in range(n):
            print("Loading from {} {}".format(f,i))
            if f.endswith(".npy"):
                _,theta_source,_,phi_source=f.replace(".npy","").split("_")
                a= make_cone_density(theta_source, phi_source,z_isgri, z_picsit, progress=False, n_events=[100,2000])
                if a is not None:
                    lock.acquire()
                    data.append(a)
                    labels.append([float(theta_source), float(phi_source)])
                    lock.release()
            if len(data)%100==0:
                print("Aquiring lock")
                lock.acquire()
                if os.path.exists(data_filename):
                    x,y = pkl.load(open(data_filename, "wb"))
                    new_x, new_y = np.array(list(data)), np.array(list(labels))
                    x,y = np.concatenate((x, new_x), axis=0), np.concatenate((y, new_y), axis=0)
                else:
                    x, y = np.array(list(data)), np.array(list(labels))

                pkl.dump((x,y), open(data_filename, "wb"))
                data.clear()
                labels.clear()
                lock.release()
                print("Realeased lock")
    
    
    with Pool(3,maxtasksperchild=10) as p:
        for t in p.imap(get_data, os.listdir("save_Compton"), chunksize=365):
            pass
    x,y = pkl.load(open(data_filename, "wb"))
    new_x, new_y = np.array(list(data)), np.array(list(labels))
    x,y = np.concatenate((x, new_x), axis=0), np.concatenate((y, new_y), axis=0)

    pkl.dump((x,y), open(data_filename,"wb"))

    print("Done generating data, save in {}. Exiting now.".format(data_filename))
    exit()
    data=list(data)
    labels=[e[0] for e in data]
    data = [e[1] for e in data]
else:
    data = pkl.load(open(data_filename, "rb"))
    labels=[e[0] for e in data]
    data = [e[1] for e in data]

# clear nones from data
usable_labels,usable_data = [],[]
for i in range(len(data)):
    if data[i] is not None:
        usable_labels.append(labels[i])
        usable_data.append(data[i])
data = [x.reshape(180,45,1) for x in usable_data]
labels = [[float(y[0]), float(y[1])] for y in usable_labels]

# create the train and test data
from sklearn.model_selection import train_test_split
import tensorflow as tf
y_train,y_test,x_train,x_test = train_test_split(labels, data, shuffle=True)
y_train,y_test = np.array(y_train), np.array(y_test)
x_train,x_test = np.array(x_train), np.array(x_test)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# convert to tensorflow tensors
y_train = tf.convert_to_tensor(y_train)
x_train = tf.convert_to_tensor(x_train)
#y_test= tf.convert_to_tensor(y_test)
#x_test= tf.convert_to_tensor(x_test)
print("Train data : {} {}".format(x_train.shape, y_train.shape))
print("Test data  : {} {}".format(x_test.shape, y_test.shape))


# define a first simple CNN network
from tensorflow.keras.layers import Conv2D, BatchNormalization, Flatten, Dense
def get_model():
    model = tf.keras.Sequential()
    model.add(Conv2D(32,3,input_shape=(180,45,1), activation="relu"))
    model.add(BatchNormalization())
    model.add(Conv2D(64,3, activation="relu"))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(512,activation="relu"))
    model.add(BatchNormalization())
    model.add(Dense(128,activation="relu"))
    model.add(BatchNormalization())
    model.add(Dense(2,activation="relu"))
    return model

# get the model
model = get_model()

epochs = 10
batch_size = 12

callbacks = [
    #keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),
    tf.keras.callbacks.TensorBoard('./logs', update_freq=1)

]
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="mean_squared_error",
    metrics=["mean_squared_error","mean_absolute_error"],
)
model.fit(
    x_train, y_train, steps_per_epoch=2,batch_size=batch_size, epochs=epochs, callbacks=callbacks, validation_data=(x_test,y_test),
)

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

