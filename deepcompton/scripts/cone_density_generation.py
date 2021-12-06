import matplotlib.pyplot as plt
import numpy as np
import os
import pickle as pkl
from deepcompton.cones import make_cone_density

plt.rcParams.update({"font.size":14})
Ee = 511

z_isgri = 0
z_picsit = -8.68

# get cone density data for all files in dataset
from multiprocessing import Pool, Manager
manager = Manager()
labels,data=manager.list(),manager.list()

def get_data(f):
    print("Loading from {}".format(f))
    if f.endswith(".npy"):
        _,theta_source,_,phi_source=f.replace(".npy","").split("_")
        labels.append([theta_source, phi_source])
        data.append(make_cone_density(theta_source, phi_source,z_isgri, z_picsit, progress=False))


with Pool(maxtasksperchild=10) as p:
    for t in p.imap(get_data, os.listdir("save_Compton"), chunksize=365):
        pass



pkl.dump((list(labels),list(data)), open("cone_density_data_full.pkl","wb"))

exit()


density = make_cone_density(theta_source, phi_source)
print(density.shape)

actual = np.radians(np.linspace(0, 360, 180))
expected = np.arange(0, 90, 2)
 
r_g, theta_g = np.meshgrid(expected,actual)
 
fig, ax = plt.subplots(figsize = (12,9),subplot_kw=dict(projection='polar'))
cax = ax.pcolormesh(theta_g, r_g, density ,cmap = "hot")
cbar = fig.colorbar(cax,label = "Nombre d'intersections de cônes")
ax.scatter(np.radians(phi_source),theta_source,label = "Position simulée")

tt = ax.get_yticklabels()
list_tt = np.linspace(90/np.size(tt),90,np.size(tt))
for i in range(np.size(tt)):
    tt[i].set_text(str(int(list_tt[i]))+"°")
    tt[i].set_color("grey")
    tt[i].set_fontweight(900)
ax.set_yticklabels(tt)
plt.legend()
plt.show()
plt.grid()
plt.tight_layout()
