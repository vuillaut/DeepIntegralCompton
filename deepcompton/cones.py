import numpy as np
import os
from astropy.table import Table
from tqdm.auto import tqdm
import pickle

from .utils import load_data
from .import utils as compton
from .progress import printProgressBar
from . import constants

from astropy.io.misc.hdf5 import write_table_hdf5

from sklearn.model_selection import train_test_split


def make_cone(x, z_isgri=constants.x_isgri, z_picsit=constants.x_picsit, Ee=constants.electron_mass):
    """
    Single cone reconstruction

    :param x: `numpy.ndarray`
        (x, y, z)
    :param z_isgri: float
    :param z_picsit: float
    :param Ee: float
    :return:
        theta, phi, cotheta
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
        cotheta = compton.cottheta(energ1cur, E0 - energ1cur)

        A = np.array([x1cur, y1cur, z1cur])
        B = np.array([x2cur, y2cur, z2cur])

        theta = compton.colatitudeaxe(B, A)
        phi = compton.longitudeaxe(B, A)

        return np.array([theta, phi, cotheta])
    else:
        return None


def make_cone_density(theta_source, phi_source, z_isgri, z_picsit, precision=5000., density_precision=2., r=1e14,
                      max_cones=2000000, lon_max=360., lat_max=90., progress=True, datadir=None, n_events=None):
    if datadir is None:
        name = "./save_Compton/theta_" + str(theta_source) + "_phi_" + str(phi_source) + ".npy"
    else:
        name = "{}/theta_".format(datadir) + str(theta_source) + "_phi_" + str(phi_source) + ".npy"

    X = np.load(name).astype(np.float64)
    N = X.shape[0]
           
    # if empty data return None
    if N == 0:
        return None

    if n_events is not None:
        if isinstance(n_events, list):
            # randomly select number of events to use
            n = np.random.choice(range(n_events[0], n_events[1]+1))
            pos = np.random.choice(range(N), size=n)
            X = X[pos]
            N = n
 
    # density grid
    density = np.zeros((int(lon_max / density_precision), int(lat_max / density_precision)))

    # cone counter
    ncones = 0

    # position des capteurs
    x1cur = z_isgri
    x2cur = z_picsit


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
                y1cur = X[i, 7]
                z1cur = -X[i, 6]
                y2cur = X[i, 9]
                z2cur = -X[i, 8]

                #colat = compton.colatconer(r, x1cur, y1cur, z1cur, theta, phi, cotheta, precision)
                #longit = compton.longitconer(r, x1cur, y1cur, z1cur, theta, phi, cotheta, precision)

                colat, longit = compton.coner(r, x1cur, y1cur, z1cur, theta, phi, cotheta, precision)

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


class AnglesDataset:
    """:arg
    """

    def __init__(self):
        pass

    def generate(self, src_dir):
        """
        generate dataset from source directory

        :param src_dir: path
        :return: `astropy.table`
        """
        filenames = [os.path.join(src_dir, f) for f in os.listdir(src_dir)]
        src_thetas = []
        src_phis = []
        thetas = []
        phis = []
        cothetas = []
        for filename in tqdm(filenames):
            try:
                data, src_theta, src_phi = load_data(filename)
            except ValueError:
                print(f"no data for {filename}")
                continue
            try:
                theta, phi, cotheta = np.apply_along_axis(make_cone, axis=1, arr=data).T
                mask = np.isfinite(theta) & np.isfinite(phi) & np.isfinite(cotheta)
                theta = theta[mask]
                phi = phi[mask]
                cotheta = cotheta[mask]

                src_thetas.append(src_theta)
                src_phis.append(src_phi)
                thetas.append(theta)
                phis.append(phi)
                cothetas.append(cotheta)

            except:
                print(f"fail for {filename}")

        self.tab = Table(data=[src_thetas, src_phis, thetas, phis, cothetas],
                         names=['src_theta', 'src_phi', 'theta', 'phi', 'cotheta'],
                         )
        return self.tab

    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self.tab, file)

    def load(self, filename):
        with open(filename, 'rb') as file:
            self.tab = pickle.load(file)
        return self.tab

    def split_tab_train_test(self, test_size=0.5, random_state=None, shuffle=True, stratify=None):
        """
        return two tables: train/test
        :return:
        """
        train_tab = Table(names=self.tab.colnames)
        test_tab = Table(names=self.tab.colnames)
        for row in self.tab:
            train_theta, test_theta, train_phi, test_phi, train_cotheta, test_cotheta = \
                train_test_split(row['theta'], row['phi'], row['cotheta'],
                                 test_size=test_size, random_state=random_state, shuffle=shuffle, stratify=stratify)
            train_tab.add_row({'src_theta': row['src_theta'],
                               'src_phi': row['src_phi'],
                               'theta': train_theta,
                               'phi': train_phi,
                               'cotheta': train_cotheta
                               })

            train_tab.add_row({'src_theta': row['src_theta'],
                               'src_phi': row['src_phi'],
                               'theta': test_theta,
                               'phi': test_phi,
                               'cotheta': test_cotheta
                               })

        # self.train_tab = train_tab
        # self.test_tab = test_tab
        train_ad = AnglesDataset()
        train_ad.tab = train_tab
        test_ad = AnglesDataset()
        test_ad.tab = test_tab
        return train_ad, test_ad


    def extend(self, value=0):
        if 'tab' not in self.__dict__.keys():
            raise AttributeError("You must load or generate the base table first")

        lengths = np.array([len(t['theta']) for t in self.tab])
        selected_rows = self.tab[lengths > 1000]
        cols = ['theta', 'phi', 'cotheta']
        redim_cols = {col: [] for col in selected_rows.colnames}
        redim_cols['src_theta'] = selected_rows['src_theta']
        redim_cols['src_phi'] = selected_rows['src_phi']
        for row in selected_rows:
            for col in cols:
                redim_cols[col].append(np.concatenate([row[col], value*np.ones(lengths.max() - len(row[col]))]))

        for name, col in redim_cols.items():
            redim_cols[name] = np.array(col)

        self.tab_extended = Table(data=redim_cols)

    def save_extended(self, filename='extended_angles.h5', **kwargs):
        """
        save in HDF5 format

        :param filename:
        :kwargs kwargs for `astropy.io.misc.hdf5.write_table_hdf5`
        """
        kwargs.setdefault('compression', True)
        kwargs['path'] = 'angles'
        write_table_hdf5(self.tab_extended, filename, **kwargs)

    def load_extended(self, filename='extended_angles.h5'):
        self.tab_extended = Table.read(filename, path='angles')

    def golden(self, min_size=1000):
        if 'tab' not in self.__dict__.keys():
            raise AttributeError("You must load or generate the base table first")

        ## compute lengths to throw the too short ones
        lengths = np.array([len(t['theta']) for t in self.tab])
        selected_rows = self.tab[lengths >= min_size]
        ## recompute lengths to get the min
        lengths = np.array([len(t['theta']) for t in selected_rows])

        cols = ['theta', 'phi', 'cotheta']
        redim_cols = {col: [] for col in selected_rows.colnames}
        redim_cols['src_theta'] = selected_rows['src_theta']
        redim_cols['src_phi'] = selected_rows['src_phi']
        for row in selected_rows:
            for col in cols:
                redim_cols[col].append(row[col][:lengths.min()])

        for name, col in redim_cols.items():
            redim_cols[name] = np.array(col)

        self.tab_gold = Table(data=redim_cols)

    def save_golden(self, filename='gold_angles.h5', **kwargs):
        """
        save in HDF5 format

        :param filename:
        :kwargs kwargs for `astropy.io.misc.hdf5.write_table_hdf5`
        """
        kwargs.setdefault('compression', True)
        kwargs['path'] = 'angles'
        write_table_hdf5(self.tab_gold, filename, **kwargs)

    def load_golden(self, filename='gold_angles.h5'):
        self.tab_gold = Table.read(filename, path='angles')



