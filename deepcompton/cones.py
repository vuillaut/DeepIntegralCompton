import numpy as np

import deepcompton.utils as compton
from deepcompton.progress import printProgressBar
from deepcompton import constants
from deepcompton.utils import get_test_data_path, load_data


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
        cotheta = compton.cottheta(energ1cur, energ2cur)

        A = np.array([x1cur, y1cur, z1cur])
        B = np.array([x2cur, y2cur, z2cur])

        theta = compton.colatitudeaxe(B, A)
        phi = compton.longitudeaxe(B, A)

        return np.array([theta, phi, cotheta])
    return None


def make_cone_density(filename=get_test_data_path(), z_isgri=constants.x_isgri, z_picsit=constants.x_picsit,
                      precision=constants.precision, density_precision=constants.density_precision,
                      r=constants.r_infinite, max_cones=2000000, lon_max=360., lat_max=90., progress=True):

    data, theta_source, phi_source = load_data(filename)

    N = len(data)
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
    for i, row in data.iterrows():
        # while cone count is not reached
        if ncones < max_cones:
            cone = make_cone(X[i, :], z_isgri, z_picsit)
            if cone is not None:
                [theta, phi, cotheta] = cone
                x1cur = z_isgri
                x2cur = z_picsit

                y1cur = row['y1']
                z1cur = -row['x1']
                y2cur = row['y2']
                z2cur = -row['x2']

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
