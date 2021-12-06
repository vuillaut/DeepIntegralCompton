import matplotlib.pyplot as plt
import numpy as np
import deepcompton.utils as compton
from deepcompton import constants

plt.rcParams.update({"font.size":14})
Ee = constants.electron_mass

z_isgri = constants.x_isgri
z_picsit = constants.x_picsit

theta_source = 42
phi_source = 104
name = compton.get_test_data_path()


def make_cones(x):
    pos1x = x[:,6]
    pos1y = x[:,7]
    energ1 = x[:,0]
    pos2x = x[:,8]
    pos2y = x[:,9]
    energ2 = x[:,1]
    
    sumenerg = energ1 + energ2
    
    spectre = np.histogram(sumenerg, bins = 2000,range = (0,2000))

def make_cone(x, z_isgri, z_picsit):
    """Single cone
    """
    x1cur = z_isgri
    x2cur = z_picsit

    energ1cur = x[0]
    energ2cur = x[1]

    y1cur = x[7]
    z1cur = -x[6]
    
    y2cur = x[9]
    z2cur = -x[8]
    

    E0 = energ1cur + energ2cur
    
    Ec = E0/(1+2*E0/Ee)
    
    if (energ2cur >= Ec) and (energ1cur <= E0 - Ec):
        E2 = E0 - energ1cur
        cotheta = compton.cottheta(energ1cur,E0 - energ1cur)
        A = np.array([x1cur,y1cur,z1cur])
        B = np.array([x2cur,y2cur,z2cur])
    
        theta = compton.colatitudeaxe(B,A)
        phi = compton.longitudeaxe(B,A)

        colat = compton.colatconer(r, x1cur, y1cur, z1cur, theta, phi, cotheta, precision) 
        longit = compton.longitconer(r, x1cur, y1cur, z1cur, theta, phi, cotheta, precision)
            

extraction = np.load(name).astype("float64")

pos1x = extraction[:,6]
pos1y = extraction[:,7]
energ1 = extraction[:,0]
pos2x = extraction[:,8]
pos2y = extraction[:,9]
energ2 = extraction[:,1]

sumenerg = energ1 + energ2

spectre = np.histogram(sumenerg, bins = 2000,range = (0,2000))

s = np.size(pos1x)

# mesurement precision
precisiondensite = constants.precisiondensite
precision = constants.precision

# density grid
densite = np.zeros((int(360/precisiondensite),int(90/precisiondensite)))

# ??
r = constants.r_infinite

# cone counter
ncones = 0

# for each row in the data create a cone
for i in range(s):
    # while cone count is not reached
    if ncones < 2000000:
        x1cur = z_isgri
        y1cur = pos1y[i]
        z1cur = -pos1x[i]
        
        x2cur = z_picsit
        y2cur = pos2y[i]
        z2cur = -pos2x[i]
        
        energ1cur = energ1[i]
        energ2cur = energ2[i]

        E0 = energ1cur + energ2cur
        
        Ec = E0/(1+2*E0/Ee)
        
        if (energ2cur >= Ec) and (energ1cur <= E0 - Ec):
      
            E2 = E0 - energ1cur
            cotheta = compton.cottheta(energ1cur,E0 - energ1cur)
            A = np.array([x1cur,y1cur,z1cur])
            B = np.array([x2cur,y2cur,z2cur])
    
            theta = compton.colatitudeaxe(B,A)
            phi = compton.longitudeaxe(B,A)
            print(theta, phi)
            
            colat = compton.colatconer(r, x1cur, y1cur, z1cur, theta, phi, cotheta, precision) 
            longit = compton.longitconer(r, x1cur, y1cur, z1cur, theta, phi, cotheta, precision)
            
            hemisphere = (colat < 90)
            longit = longit[hemisphere]
            colat = colat[hemisphere]
            d = np.zeros((int(360./precisiondensite), int(90./precisiondensite)))
            
            l = ((abs(longit)%360)/precisiondensite).astype(int)
            c = ((abs(colat)%90)/precisiondensite).astype(int)
    
            d[l,c]=1.
    
            densite = densite + d
            
            ncones += 1
            break
            

    
        if(i%100==0):
            print(i)
        
        
# fig = plt.figure()
# ax = fig.add_subplot(111,projection = "hammer")
# cax = plt.imshow(np.transpose(densite), cmap ="hot",extent = [-90,90 + 180,-90,90], interpolation = None)
# plt.xlabel("Longitude Phi (degrés)")
# plt.ylabel("Colatitude Theta (degrés)")
# plt.title("Données INTEGRAL")


# plt.grid()

# cbar = fig.colorbar(cax, label = "Nombre d'intersections de cônes")
# plt.show()

actual = np.radians(np.linspace(0, 360, 180))
expected = np.arange(0, 90, 2)
 
r_g, theta_g = np.meshgrid(expected,actual)
values = densite
 
fig, ax = plt.subplots(figsize = (12,9),subplot_kw=dict(projection='polar'))
# cax = ax.contourf(theta_g, r_g, values,cmap = "hot")
cax = ax.pcolormesh(theta_g, r_g, values,cmap = "hot")
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
