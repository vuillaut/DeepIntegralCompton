# Deep Learning applique a l'imagerie Compton avec les donnees du satellite INTEGRAL
# Hackatlon AstroInfo 2021


##___ Importations

import numpy as np
import matplotlib.pyplot as plt

import Utilitaires_Compton as compton
import pickle as pkl
from os import listdir
from sys import setrecursionlimit


# ___ Constantes

Ee = 511
z_isgri = 0 # position plan ISGRI -> 0
z_picsit = -8.68  # position plan PICSIT -> distance entre les deux plans
r = 100000000000000.
precision = 5000.



## ___ Fonctions

def angular_separation(colat1, long1, colat2, long2):
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

    cosdelta = np.sin(colat1) * np.sin(colat2) * np.cos(
        (long1 - long2)) + np.cos(colat1) * np.cos(colat2)

    cosdelta[cosdelta > 1] = 1.
    cosdelta[cosdelta < -1] = -1.

    ang_sep = np.arccos(cosdelta)
    return ang_sep


def AddFunction_DistCone(f1,f2):
    """Somme de deux fonctions produisant une distribution 
    pour un ou plusieurs cones.
    """
    return lambda l,c : f1(l,c) + f2(l,c)


def Cone_param(pos1x,pos1y,energ1,pos2x,pos2y,energ2) :
    """ Retourne les parametres theta, phi et cottheta d'un cone
    a partir des observations pos1x,pos1y,energ1,pos2x,pos2y,energ2.
    retourne None si l'energie limite est atteinte.
    -----------
    Attention :
    indice 1 : ISGRI, indice 2 : PICsIT
    """
    x1cur = z_isgri
    y1cur = pos1y
    z1cur = -pos1x
    
    x2cur = z_picsit
    y2cur = pos2y
    z2cur = -pos2x
    
    E0 = energ1 + energ2
    
    Ec = E0/(1+2*E0/Ee) 
    
    if (energ2 >= Ec) and (energ1 <= E0 - Ec):
        cotheta = compton.cottheta(energ1,E0 - energ1)
        A = np.array([x1cur,y1cur,z1cur])
        B = np.array([x2cur,y2cur,z2cur])

        theta = compton.colatitudeaxe(B,A)
        phi = compton.longitudeaxe(B,A)
        
        return theta, phi, cotheta
    
    else :
        return None

    
def Normal(delta_x,sigma):
    """Loi normale
    """
    return 1/(sigma * np.sqrt(np.pi)) * np.exp(-1/2*(delta_x /sigma)**2)


def DistriCone_v3_2D(theta, phi, cottheta, sigma ):
    """ Retourne la distribution associee au cone de parametres theta, phi, cottheta.
    c'est-a-dire le cone plus une incertitude gaussienne de parametre sigma.
    """
    
    def f_2D(l,c) :
        """l et c 2D
        output 2D
        """
        theta_list = np.full_like(c, theta)
        phi_list = np.full_like(l, phi)
        
        e = angular_separation(theta_list, phi_list, 
                               np.radians(c), np.radians(l))
        delta = e - np.arctan(1/cottheta)
        return Normal(delta,np.radians(sigma))
    
    return f_2D



def Create_image(data, n_cones, sigma = 2, grid_longit = np.linspace(0, 359, 180), grid_colat = np.linspace(0, 89, 45) ):
    """ A partir des data simulees ou observees, creation d'une image sur la grille (grid_longit, grid_colat).
    Pour cela on applique sur cette grille la fonction cumulant n_cones distributions produites avec n_cones evenements
    de data, tires aleatoirement. Ces distributions gaussiennes ont une incertitude sigma.
    """
    
    # ___ Load data
    s = data.shape[0]
    # ISGRI -> 1
    E1_list = data[:,0]
    Y1_list = data[:,6]
    Z1_list = data[:,7]
    #PICsIT -> 2
    E2_list = data[:,1]
    Y2_list = data[:,8]
    Z2_list = data[:,9]
    
    # ___ Create distribution
    f_v3 = lambda l,c : 0 

    for i in np.random.randint(0,s,size = n_cones) :
        L = Cone_param(Y1_list[i],Z1_list[i],E1_list[i],Y2_list[i],Z2_list[i],E2_list[i])
        if L!= None :
            theta, phi, cottheta = L
            f_new = DistriCone_v3_2D(theta, phi, cottheta, sigma = sigma)
            f_v3 = AddFunction_DistCone(f_v3,f_new)

    # ___ Application on image 
    l, c = np.meshgrid(grid_colat, grid_longit)
    im = f_v3(c,l)
    
    return im


def Create_dataset(name_file_list,path, name_pkl, n_file_max = -1, n_image_file = 50, n_cones_min = 100, n_cones_max = 2000, sigma = 2):
    """ Creation d'un dataset a partir d'une liste de fichiers path+name_file_list.
    A partir de n_file_max fichiers, on tire n_image_file images par fichier 
    avec un nombre de cones variant aleatoirement entre n_cones_min et n_cones_max.
    Le dataset est enregistre dans un pkl nomme name_pkl.
    """
    y,x=[],[]
    i=0
    for name_file in name_file_list[:n_file_max] :
        i+=1
        _,theta,_,phi=name_file.replace(".npy","").split("_")
        j=0
        for n_cones in np.random.randint(n_cones_min,n_cones_max+1,size = n_image_file) :
            data = np.load(path+name_file).astype("float64")
            if len(data)>0:
                #print(f"\r file n°{i}/{n_file_max}, image n°{j}/{n_image_file} ", end='', flush=True)
                data =  Create_image(data, n_cones, sigma = sigma)
                y.append([float(theta),float(phi),n_cones])
                x.append(data)
                j+=1
            if len(y)%100 == 0:
                nx,ny = pkl.load(open(name_pkl, "rb"))
                nx+=x
                ny+=y
                pkl.dump((nx,ny), open(name_pkl, "wb"))
                x.clear()
                y.clear()
    nx,ny = pkl.load(open(name_pkl, "rb"))
    nx+=x
    ny+=y
    pkl.dump((nx,ny), open(name_pkl, "wb"))
    return 0




## ___ Script

path = '/Users/lg265853/Documents/AstroInfo/save_Compton'
name_file_list = listdir(path)

n_file_max = -1
n_image_file = 100
n_cones_min = 100
n_cones_max = 2000
sigma = 2
name_pkl = f'dataset_{n_image_file}images_{n_file_max}files_sig{sigma}_ncones{n_cones_min}-{n_cones_max}.pkl'


setrecursionlimit(n_cones_max)

Create_dataset(name_file_list,path+'/',name_pkl,n_file_max, n_image_file, n_cones_min, n_cones_max, sigma)




## ___ Verification

#x,y = pkl.load(open(name_pkl, "rb")) 
#print(len(x), len(y))
#
#plt.figure(figsize=(20,20))
#k = 1
#for i in np.random.randint(0,len(x),size = 30):
#    plt.subplot(3,10,k)
#    plt.imshow(x[i], origin='lower', cmap = 'CMRmap_r')
#    plt.scatter(y[i][0]/2,y[i][1]/2)
#    plt.title(str(y[i][2]) + ' cones')
#    plt.xlabel('colatitude') ; plt.ylabel('longitude')         
#    k+=1
#
#plt.show()
