from deepcompton import utils
import numpy as np

def test_get_test_data_path():
    path = utils.get_test_data_path()
    assert path.endswith('theta_42_phi_104.npy')


def test_load_data():
    path = utils.get_test_data_path()
    data = utils.load_data(path)
    assert len(data) > 0


def test_angular_separation():
    seps = utils.angular_separation(np.array([0, 0]), np.array([1.42, 0]),
                                    np.array([0, np.pi / 2.]), np.array([42.1, 0]),
                                    )
    np.testing.assert_allclose(seps, [0., np.pi / 2])

