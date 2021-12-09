# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 13:45:36 2021

@author: gdaniel
"""

from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Conv3D, Input, Flatten, AveragePooling3D,\
                             MaxPooling3D, BatchNormalization, Activation, Dropout,LeakyReLU,\
                             UpSampling1D, merge
from keras import callbacks
from keras import optimizers
from keras import regularizers
import tensorflow as tf
import numpy as np
import keras.losses

plt.rcParams.update({"font.size":16})
dic = np.load("dic_theta_phi_delta.npy",allow_pickle = True)[0]

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


def create_data_test(batch_size):

    X_val = np.zeros((batch_size,size_th,size_phi,size_delta,1))
    
    Y_val = np.zeros((batch_size,2))
        
    for i in range(batch_size):
    
        index = np.random.randint(size_keys)
        
        key_cur = keys[index]
        
        list_evnt = dic[key_cur]
        
        n_cones = np.random.randint(nmin_cones,nmax_cones)
        
        size_list_evnt = np.shape(list_evnt)[0]
        
        index_evnt_sel = np.random.randint(size_list_evnt,size = n_cones)
        
        list_evnt_sel = list_evnt[index_evnt_sel]
        
        hist_3d,bx = np.histogramdd(list_evnt_sel,bins = [size_th,size_phi,size_delta],range = [(0,max_th),(0,max_phi),(0,max_delta)])
        
        X_val[i,:,:,:,0] = hist_3d/np.sum(hist_3d)
        
        Y_val[i,0] = (key_cur[0])/90
        
        Y_val[i,1] = (key_cur[1])/360
                    
    return X_val,Y_val

def angular_separation(y_true,y_pred):
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
    
    colat1 = y_true[:,0]*np.pi/2
    long1 = y_true[:,1]*np.pi*2
    colat2 = y_pred[:,0]*np.pi/2
    long2 = y_pred[:,1]*np.pi*2
    
    cosdelta = tf.math.sin(colat1) * tf.math.sin(colat2) * tf.math.cos(
        (long1 - long2)) + tf.math.cos(colat1) * tf.math.cos(colat2)

    # cosdelta = cosdelta*(cosdelta <= 1)*(cosdelta >= -1) + (cosdelta < -1) + (cosdelta > 1)
    ang_sep = tf.math.acos(cosdelta)*180/np.pi
    return ang_sep

def angular_separation_np(y_true,y_pred):
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
    
    colat1 = y_true[:,0]*np.pi/2
    long1 = y_true[:,1]*np.pi*2
    colat2 = y_pred[:,0]*np.pi/2
    long2 = y_pred[:,1]*np.pi*2
    
    cosdelta = np.sin(colat1) * np.sin(colat2) *np.cos(
        (long1 - long2)) + np.cos(colat1) * np.cos(colat2)

    # cosdelta = cosdelta*(cosdelta <= 1)*(cosdelta >= -1) + (cosdelta < -1) + (cosdelta > 1)
    ang_sep = np.arccos(cosdelta)*180/np.pi
    return ang_sep

def cos_angular_separation(y_true,y_pred):
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
    
    colat1 = y_true[:,0]*np.pi/2
    long1 = y_true[:,1]*np.pi*2
    colat2 = y_pred[:,0]*np.pi/2
    long2 = y_pred[:,1]*np.pi*2
    
    cosdelta = tf.math.sin(colat1) * tf.math.sin(colat2) * tf.math.cos(
        (long1 - long2)) + tf.math.cos(colat1) * tf.math.cos(colat2)

    # cosdelta = cosdelta*(cosdelta <= 1)*(cosdelta >= -1) + (cosdelta < -1) + (cosdelta > 1)
    return -cosdelta
# keras.losses.custom_loss = cos_angular_separation

model = load_model("model_1.hdf5", custom_objects={'cos_angular_separation':cos_angular_separation,'angular_separation':angular_separation})


theta_source = 65
phi_source = 210

tab = np.load("real_data_extracted_theta_"+str(theta_source)+"_phi_"+str(phi_source)+"_tab.npy")
# tab = np.load("theta_38_phi_300_tab.npy")


hist_3d,bx = np.histogramdd(tab,bins = [size_th,size_phi,size_delta],range = [(0,max_th),(0,max_phi),(0,max_delta)])
hist_3d = hist_3d/np.sum(hist_3d)        
X_test = np.expand_dims(hist_3d,axis = (0,4))

Y_pred = model.predict(X_test)

Y_test = np.array([[theta_source/90,phi_source/360]])

ang_sep = angular_separation_np(Y_test,Y_pred)

plt.figure(figsize = (16,9))
plt.hist(ang_sep,bins = 180,range = (0,180))
plt.xlabel("Angular separation")
plt.ylabel("Counts")
plt.title("Mean angular separation: " + str(np.round(np.mean(ang_sep),1)) + " degrees")


i = np.random.randint(batch_test)
y_pred_cur = Y_pred[i]
y_test_cur = Y_test[i]

theta_pred = y_pred_cur[0]*90
phi_pred = y_pred_cur[1]*360

if theta_pred < 0:
    theta_pred = -theta_pred
    phi_pred = 180 + phi_pred

theta_test = y_test_cur[0]*90
phi_test = y_test_cur[1]*360

r_g, theta_g = np.meshgrid(expected,actual)
 
fig, ax = plt.subplots(figsize = (12,9),subplot_kw=dict(projection='polar'))
ax.set_ylim(0,90)

# cax = ax.contourf(theta_g, r_g, values,cmap = "hot")

ax.scatter(np.radians(phi_test),theta_test,label = "Position simulée")
ax.scatter(np.radians(phi_pred),theta_pred,label = "Position prédite")
tt = ax.get_yticklabels()
list_tt = np.linspace(90/np.size(tt),90,np.size(tt))
for i in range(np.size(tt)):
    tt[i].set_text(str(int(list_tt[i]))+"°")
    tt[i].set_color("grey")
    tt[i].set_fontweight(900)
ax.set_yticklabels(tt)
ax.set_title("Number of cones: "+str(np.sum(X_test[i])))
plt.legend()
plt.grid()
plt.show()

plt.tight_layout()