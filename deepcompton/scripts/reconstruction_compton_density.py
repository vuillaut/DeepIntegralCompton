import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import pkg_resources

import deepcompton.utils as compton
from deepcompton import constants
from deepcompton import vizualisation as viz

def main(theta_source=42, phi_source=104):

    plt.rcParams.update({"font.size": 14})
    Ee = constants.electron_mass
    z_isgri = constants.x_isgri
    z_picsit = constants.x_picsit


    name = pkg_resources.resource_filename('deepcompton', f'data/theta_{theta_source}_phi_{phi_source}.npy')
    name = "save_Compton/theta_{}_phi_{}.npy".format(theta_source, phi_source)
    if not os.path.exists(name):
        print("File {} not found. Exiting".format(name))
        exit()
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

    precisiondensite = constants.precisiondensite
    precision = constants.precision
    r = constants.r_infinite

    densite = np.zeros((int(360/precisiondensite),int(90/precisiondensite)))


    ncones = 0

    indexes = []
    cothetas = []
    phis = []
    thetas = []

    for i in range(s):

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

                colat = compton.colatconer(r, x1cur, y1cur, z1cur, theta, phi, cotheta, precision)
                longit = compton.longitconer(r, x1cur, y1cur, z1cur, theta, phi, cotheta, precision)

                indexes.append(i)
                cothetas.append(cotheta)
                thetas.append(theta)
                phis.append(phi)

                hemisphere = (colat < 90)
                longit = longit[hemisphere]
                colat = colat[hemisphere]
                d = np.zeros((int(360./precisiondensite), int(90./precisiondensite)))

                l = ((abs(longit)%360)/precisiondensite).astype(int)
                c = ((abs(colat)%90)/precisiondensite).astype(int)

                d[l,c]=1.

                densite = densite + d

                ncones += 1



            #if(i%100==0):
            #    print(i)


    np.savetxt(f'reco_theta_{theta_source}_phi_{phi_source}.txt',
               np.transpose([indexes, cothetas, thetas, phis]),
               header='index cotheta theta phi',
               fmt='%.4e'
               )


    actual = np.radians(np.linspace(0, 360, 180))
    expected = np.arange(0, 90, 2)

    r_g, theta_g = np.meshgrid(expected, actual)
    print("sum : ", np.sum(densite))

    from deepcompton.cones import make_cone_density
    density = make_cone_density(theta_source, phi_source, z_isgri, z_picsit)
    print("msum : ", np.sum(density))

    ax = viz.plot_backprojected(theta_g, r_g, densite)
    ax = viz.plot_source_pos(theta_source, np.radians(phi_source), ax=ax)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # if user has provided a source theta and phi value use it, otherwise use defaults
    if len(sys.argv)==3:
        theta_source = int(sys.argv[1])
        phi_source = int(sys.argv[2])
    else:
        theta_source = 42
        phi_source = 104

    main(theta_source, phi_source)

