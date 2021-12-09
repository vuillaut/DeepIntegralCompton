# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 13:45:36 2021

@author: gdaniel
"""

from keras.models import Sequential, Model
from keras.layers import Dense, Conv3D, Input, Flatten, AveragePooling3D, \
    MaxPooling3D, BatchNormalization, Activation, Dropout, LeakyReLU, \
    UpSampling1D, merge
from keras import callbacks
from keras import optimizers
from keras import regularizers
import tensorflow as tf
import numpy as np
import matplotlib .pyplot as plt

dic = np.load("dic_theta_phi_delta.npy", allow_pickle=True)[0]

keys = (list(dic.keys()))

size_keys = np.shape(keys)[0]

nmin_cones = 100
nmax_cones = 2000

size_th = 45
size_phi = 180
size_delta = 90

max_th = 90
max_phi = 360
max_delta = 180


def create_data(batch_size):
    X_val = np.zeros((batch_size, size_th, size_phi, size_delta, 1))

    Y_val = np.zeros((batch_size, 2))

    for i in range(batch_size):
        index = np.random.randint(size_keys)

        key_cur = keys[index]

        list_evnt = dic[key_cur]

        n_cones = np.random.randint(nmin_cones, nmax_cones)

        size_list_evnt = np.shape(list_evnt)[0]

        index_evnt_sel = np.random.randint(size_list_evnt, size=n_cones)

        list_evnt_sel = list_evnt[index_evnt_sel]

        hist_3d, bx = np.histogramdd(list_evnt_sel, bins=[size_th, size_phi, size_delta],
                                     range=[(0, max_th), (0, max_phi), (0, max_delta)])

        X_val[i, :, :, :, 0] = hist_3d / np.sum(hist_3d)

        Y_val[i, 0] = (key_cur[0]) / 90

        Y_val[i, 1] = (key_cur[1]) / 360

    return X_val, Y_val


def angular_separation(y_true, y_pred):
    """
    Compute the angular separation in radians
    between two pointing direction given with lat-long
    Parameters
    ----------
    lat1: 1d `numpy.ndarray` , latitude of the first pointing direction
    long1: 1d `numpy.ndarray` longitude of the first pointing direction
    lat2: 1d `numpy.ndarray`, latitude of the second pointing direction
    long2: 1d `numpy.ndarray`, longitude of the second pointing direction
    Returns
    -------
    1d `numpy.ndarray`, angular separation
    """
    # ang_sep = coordinates.angular_separation(long1, np.pi/2.-colat1, long2, np.pi/2 - colat2)

    colat1 = y_true[:, 0] * np.pi / 2
    long1 = y_true[:, 1] * np.pi * 2
    colat2 = y_pred[:, 0] * np.pi / 2
    long2 = y_pred[:, 1] * np.pi * 2

    cosdelta = tf.math.sin(colat1) * tf.math.sin(colat2) * tf.math.cos(
        (long1 - long2)) + tf.math.cos(colat1) * tf.math.cos(colat2)

    # cosdelta = cosdelta*(cosdelta <= 1)*(cosdelta >= -1) + (cosdelta < -1) + (cosdelta > 1)
    ang_sep = tf.math.acos(cosdelta) * 180 / np.pi
    return ang_sep


def cos_angular_separation(y_true, y_pred):
    """
    Compute the angular separation in radians
    between two pointing direction given with lat-long
    Parameters
    ----------
    lat1: 1d `numpy.ndarray` , latitude of the first pointing direction
    long1: 1d `numpy.ndarray` longitude of the first pointing direction
    lat2: 1d `numpy.ndarray`, latitude of the second pointing direction
    long2: 1d `numpy.ndarray`, longitude of the second pointing direction
    Returns
    -------
    1d `numpy.ndarray`, angular separation
    """
    # ang_sep = coordinates.angular_separation(long1, np.pi/2.-colat1, long2, np.pi/2 - colat2)

    colat1 = y_true[:, 0] * np.pi / 2
    long1 = y_true[:, 1] * np.pi * 2
    colat2 = y_pred[:, 0] * np.pi / 2
    long2 = y_pred[:, 1] * np.pi * 2

    cosdelta = tf.math.sin(colat1) * tf.math.sin(colat2) * tf.math.cos(
        (long1 - long2)) + tf.math.cos(colat1) * tf.math.cos(colat2)

    # cosdelta = cosdelta*(cosdelta <= 1)*(cosdelta >= -1) + (cosdelta < -1) + (cosdelta > 1)
    return -cosdelta


val_size = 1000

X_val, Y_val = create_data(val_size)

model = Sequential()
model.add(Conv3D(16, 5, input_shape=(size_th, size_phi, size_delta, 1), padding='same', activation='relu',
                 kernel_initializer='he_normal'))
# model.add(BatchNormalization())
model.add(MaxPooling3D(2))

model.add(Conv3D(32, 5, padding='same', activation='relu', kernel_initializer='he_normal'))
# model.add(BatchNormalization())
model.add(MaxPooling3D(2))

model.add(Conv3D(64, 5, padding='same', activation='relu', kernel_initializer='he_normal'))
# model.add(BatchNormalization())
model.add(MaxPooling3D(2))

model.add(Conv3D(128, 5, padding='same', activation='relu', kernel_initializer='he_normal'))
# model.add(BatchNormalization())
model.add(MaxPooling3D(2))

model.add(Flatten())
model.add(Dense(600, activation='relu', kernel_initializer='he_normal'))
# model.add(BatchNormalization())
# model.add(Dropout(0.2))
model.add(Dense(1100, activation='relu', kernel_initializer='he_normal'))
# model.add(Dense(1100, activation='relu',kernel_initializer = 'he_normal'))
# model.add(Dense(1100, activation='relu',kernel_initializer = 'he_normal'))
# model.add(BatchNormalization())
# model.add(Dropout(0.2))
model.add(Dense(2, activation=None, kernel_initializer='he_normal'))

print(model.summary())
opt = optimizers.adam(lr=0.00001)

model.compile(loss=cos_angular_separation, optimizer=opt, metrics=[angular_separation])

n_epochs = 10000000000

batch_size = 128

for i in range(n_epochs):

    X_train, Y_train = create_data(batch_size)

    if i % 10 == 0:
        model.fit(X_train, Y_train, epochs=1, batch_size=2, validation_data=(X_val, Y_val))
    else:
        model.fit(X_train, Y_train, epochs=1, batch_size=2)

model.save("model_1.hdf5")
