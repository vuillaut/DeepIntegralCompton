import numpy as np

import deepcompton.utils as compton
from deepcompton.progress import printProgressBar
from deepcompton import constants


def make_cone(x, z_isgri=constants.x_isgri, z_picsit=constants.x_picsit, Ee=constants.electron_mass):
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

    Ec = E0 / (1 + 2 * E0 / Ee)

    # why this condition ??
    if (energ2cur >= Ec) and (energ1cur <= E0 - Ec):
        E2 = E0 - energ1cur
        cotheta = compton.cottheta(energ1cur, E0 - energ1cur)

        A = np.array([x1cur, y1cur, z1cur])
        B = np.array([x2cur, y2cur, z2cur])

        theta = compton.colatitudeaxe(B, A)
        phi = compton.longitudeaxe(B, A)

        return np.array([theta, phi, cotheta])
    return None


def make_cone_density(theta_source, phi_source, z_isgri, z_picsit, precision=5000., density_precision=2., r=1e14,
                      max_cones=2000000, lon_max=360., lat_max=90., progress=True):
    name = "./save_Compton/theta_" + str(theta_source) + "_phi_" + str(phi_source) + ".npy"
    X = np.load(name).astype(np.float64)
    N = X.shape[0]
    # if empty data return None
    if N == 0:
        return None

    # density grid
    density = np.zeros((int(lon_max / density_precision), int(lat_max / density_precision)))

    # cone counter
    ncones = 0

    # for each row in the data create a cone
    if progress:
        progress_msg = "Loading cones, theta:{}, phi:{}".format(theta_source, phi_source)
        printProgressBar(0, N, prefix=progress_msg, suffix='Complete', length=50)
    for i in range(N):
        # while cone count is not reached
        if ncones < max_cones:
            cone = make_cone(X[i, :], z_isgri, z_picsit)
            if cone is not None:
                [theta, phi, cotheta] = cone
                x1cur = z_isgri
                x2cur = z_picsit

                y1cur = X[i, 7]
                z1cur = -X[i, 6]
                y2cur = X[i, 9]
                z2cur = -X[i, 8]

                colat = compton.colatconer(r, x1cur, y1cur, z1cur, theta, phi, cotheta, precision)
                longit = compton.longitconer(r, x1cur, y1cur, z1cur, theta, phi, cotheta, precision)

                hemisphere = (colat < 90)
                longit = longit[hemisphere]
                colat = colat[hemisphere]
                d = np.zeros((int(lon_max / density_precision), int(lat_max / density_precision)))

                l = ((abs(longit) % lon_max) / density_precision).astype(int)
                c = ((abs(colat) % lat_max) / density_precision).astype(int)

                d[l, c] = 1.

                density += d

                ncones += 1
        if progress:
            printProgressBar(i + 1, N, prefix=progress_msg, suffix='Complete', length=50)
    return density
