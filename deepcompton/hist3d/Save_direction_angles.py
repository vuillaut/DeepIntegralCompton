import matplotlib.pyplot as plt
import numpy as np
import Utilitaires_Compton as compton
from os.path import isfile, join

theta_vec = np.arange(0,92,2)
phi_vec = np.arange(0,364,4)

dic = {}

for i in range(np.size(theta_vec)):
    print(i)
    for j in range(np.size(phi_vec)):
        
        
        Ee = 511
        
        z_isgri = 0
        z_picsit = -8.68
        
        theta_source = int(theta_vec[i])
        phi_source = int(phi_vec[j])
        
        name = "./save_Compton/theta_"+str(theta_source)+"_phi_"+str(phi_source)+".npy"
        if isfile(name):
            extraction = np.load(name).astype("float64")
            
            if extraction.shape[0] != 0:
            
                pos1x = extraction[:,6]
                pos1y = extraction[:,7]
                energ1 = extraction[:,0]
                pos2x = extraction[:,8]
                pos2y = extraction[:,9]
                energ2 = extraction[:,1]
                
                sumenerg = energ1 + energ2
                
                
                spectre_tot,b = np.histogram(sumenerg, bins = 2000,range = (0,2000))
                
                # imin = 110000
                # imax = 118000
                
                # pos1x = pos1x[imin:imax]
                # pos1y = pos1y[imin:imax]
                # energ1 = energ1[imin:imax]
                # pos2x = pos2x[imin:imax]
                # pos2y = pos2y[imin:imax]
                # energ2 = energ2[imin:imax]
                
                # sumenerg = energ1 + energ2
                
                # spectre_1,b = np.histogram(sumenerg, bins = 2000,range = (0,2000))
                
                # plt.plot(b[:-1],spectre_tot/np.sum(spectre_tot))
                # plt.plot(b[:-1],spectre_1/np.sum(spectre_1))
                # stop
                # sel = sumenerg < 350
                
                # pos1x = pos1x[sel]
                # pos1y = pos1y[sel]
                # energ1 = energ1[sel]
                # pos2x = pos2x[sel]
                # pos2y = pos2y[sel]
                # energ2 = energ2[sel]
                
                
                
                s = np.size(pos1x)
                
                tab = np.zeros((s,3))
                            
                ncones = 0
                
                        
                x1cur = z_isgri
                y1cur = pos1y
                z1cur = -pos1x
                
                x2cur = z_picsit
                y2cur = pos2y
                z2cur = -pos2x
                
                energ1cur = energ1
                energ2cur = energ2
        
                E0 = energ1cur + energ2cur
                
                Ec = E0/(1+2*E0/Ee)
              
                E2 = E0 - energ1cur
                
                theta_diff = compton.theta_diff(energ1cur,E0 - energ1cur)

                
                theta = compton.colatitudeaxetab(x2cur,y2cur,z2cur,x1cur,y1cur,z1cur)
                phi = compton.longitudeaxetab(x2cur,y2cur,z2cur,x1cur,y1cur,z1cur)
                
                theta_diff_deg = np.degrees(theta_diff)
                theta_deg = np.degrees(theta)
                phi_deg = np.degrees(phi)
                
                tab[:,0] = theta_deg
                tab[:,1] = phi_deg
                tab[:,2] = theta_diff_deg          
                # dic["theta_"+str(theta_source)+"_phi_"+str(phi_source)] = tab
                dic[theta_source,phi_source] = tab

np.save("dic_theta_phi_delta.npy",np.array([dic]))
