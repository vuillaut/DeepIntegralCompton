from pathlib import Path
import pickle

class BaseDataset:
    """
    Base Dataset classs
    """

    def __init__(self):
        self.basedir = Path(__file__).parent[1].joinpath('data')

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


