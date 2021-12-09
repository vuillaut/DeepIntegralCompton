from pathlib import Path
from tqdm.auto import tqdm
import numpy as np
import pickle
import os
from astropy.table import Table
import pickle as pkl
from multiprocessing import Pool, Manager
from threading import Lock

from .cones import make_cone_density
from .utils import load_data
from .cones import make_cone
from .constants import x_isgri, x_picsit

class BaseDataset:
    """
    Base Dataset classs
    """

    def __init__(self):
        self.basedir = Path(__file__).parent.joinpath('data')

    def generate(self, src_dir):
        """
        generate dataset from source directory of *.npy filess

        :param src_dir: path
        """
        self.data = None
        raise NotImplementedError("This is only the base class, supercharge this method please")
        return self.data

    def save(self, filename='basefile.pickle'):
        self.filepath = self.basedir.joinpath(filename)
        with open(self.filepath, 'wb') as file:
            pickle.dump(self.data, file)

    def load(self, filename=None):
        if filename is not None:
            self.filepath = self.basedir.joinpath(filename)
        with open(self.filepath, 'rb') as file:
            self.data = pickle.load(file)



"""Generation of the Cone Density Dataset with a single source
"""
class SingleSourceDensityDataset:
    target_filename = "single_source_density_dataset.pkl"
    source_directory = "save_Compton"
    max_threads = 1
    n = 100

    def __init__(self, filename=None):
        if filename is not None:
            self.filename = filename
        pass

    def generate(self):
        """Create the datafile
        """
        # get cone density data for all files in dataset
        manager = Manager()
        data = manager.list()
        labels = manager.list()
        lock = Lock()

        def get_data(filename):
            for i in range(self.n):
                print("Loading from {} {}".format(filename, i))
                if filename.endswith(".npy"):
                    _, theta_source, _, phi_source = filename.replace(".npy", "").split("_")
                    lock.acquire()
                    labels.append([float(theta_source), float(phi_source)])
                    data.append(make_cone_density(theta_source, phi_source, x_isgri, x_picsit, progress=False,
                                                  n_events=[100, 2000]))
                    lock.release()
                if len(data) % 100 == 0:
                    print("Aquiring lock")
                    lock.acquire()
                    # load data already available
                    x, y = pkl.load(open(self.target_filename))
                    new_x, new_y = np.array(list(data)), np.array(list(labels))
                    x = np.concatenate((x, new_x), axis=0)
                    y = np.concatenate((y, new_y), axis=0)
                    pkl.dump((x, y), open(self.target_filename, "wb"))
                    # clear the data and label lists
                    data.clear()
                    labels.clear()
                    lock.release()
                    print("Realeased lock")

        with Pool(self.max_threads, maxtasksperchild=10) as p:
            for t in p.imap(get_data, os.listdir("save_Compton"), chunksize=365):
                pass

    @staticmethod
    def load(filename=None):
        """Load the dataset from the pickle file
        """
        if filename is not None:
            return pkl.load(open(SingleSourceDensityDataset.target_filename))
        return pkl.load(open(SingleSourceDensityDataset.target_filename))

        if __name__ == "__main__":
        # start the data generation
            dataset = SingleSourceDensityDataset(filename="hahah.pkl")
        dataset.generate()




