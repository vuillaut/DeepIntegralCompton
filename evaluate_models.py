import os
import tensorflow as tf
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from deepcompton.utils import angular_separation

realdatadir = ["SetImageReal_theta_38_phi_303.pkl", "SetImageReal_theta_65_phi_210.pkl"]
models_dir = "./models"
model_scores = {}
model_separations = {}
mean_separations = {}
# real data images are stored in a pickle file
n_cones = np.arange(100, 2000, 50)

from deepcompton.utils import angular_separation
def angular_loss(y_true, y_pred):
    return -1. * (tf.math.sin(y_true[:,0])*tf.math.sin(y_pred[:,0])*
                  tf.math.cos(y_true[:,1]-y_pred[:,1])+
                  tf.math.cos(y_pred[:,0])*tf.math.cos(y_true[:,0]))
def angle(yt,yp):
    return tf.math.acos(-1.*angular_loss(yt,yp)) * 180. / np.pi

for f in os.listdir(models_dir):
    dirpath = os.path.join(models_dir, f)
    model_name = f
    model_separations[model_name]=[]

    for filename in os.listdir(dirpath):
        if filename.endswith(".hdf5"):
            model_filename = os.path.join(dirpath, filename)
            model = tf.keras.models.load_model(model_filename, custom_objects={"angular_loss":angular_loss, "angle":angle})
            total_sep = []
            print("Model : {}".format(model_name))
            for real_filename in realdatadir:
                [_,_,theta,_,phi] = real_filename.replace(".pkl","").split("_")
                theta=int(theta)
                phi=int(phi)
                realx,_,_ = pkl.load(open(real_filename,"rb"))
                ang_sep = []
                for i in range(len(realx)):
                    y_pred = model(realx[i].reshape(1,180,45,1)).numpy()
                    y_real = np.radians(np.array([theta, phi]))
                    print(y_real , y_pred)
                    ang_sep.append(angular_separation(np.array([y_real[0]]), np.array([y_real[1]]), np.array([y_pred[0]]), np.array([y_pred[1]]))*180./np.pi)
                    total_sep+=ang_sep
                model_separations[model_name].append([theta,phi], ang_sep)
            mean_separations[m] = np.mean(total_sep)

# separation as function of number of cones for each theta, phi
plt.figure()
title_set = False
for m in model_separations:
    if not title_set:
        [theta,phi]=model_separation[m][0]
        plt.title("{} - Angular separation as function of cones\ntheta:{}, phi:{}".format(m, theta, phi))
        title_set = True
    plt.plot(n_cones, model_separations[m][0][1], label = m)
plt.xlabel("Cones")
plt.ylabel("Separation (deg)")
plt.legend()
plt.savefig("theta_{}_phi_{}_separation_cones.png".format(theta, phi))

# separation as function of number of cones for each theta, phi
plt.figure()
title_set = False
for m in model_separations:
    if not title_set:
        [theta,phi]=model_separation[m][1]
        plt.title("{} - Angular separation as function of cones\ntheta:{}, phi:{}".format(m, theta, phi))
        title_set = True
    plt.plot(n_cones, model_separations[m][1][1], label = m)
plt.xlabel("Cones")
plt.ylabel("Separation (deg)")
plt.legend()
plt.savefig("theta_{}_phi_{}_separation_cones.png".format(theta, phi))


# bar plot
models_name = list(mean_separations.keys())
means = list(mean_separations.values())
pos = np.argsort(means)

y_pos = np.arange(len(means))
plt.figure()
plt.title("Model performances")
plt.bar(y_pos, means[pos])
plt.xticks(y_pos, models_name)
plt.ylabel("Mean angular separation (deg)")
plt.savefig("model_perfs.png")
