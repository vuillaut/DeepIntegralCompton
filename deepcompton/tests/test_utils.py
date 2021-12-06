from deepcompton import utils


def test_get_test_data_path():
    path = utils.get_test_data_path()
    assert path.endswith('theta_42_phi_104.npy')

def test_load_data():
    path = utils.get_test_data_path()
    data = utils.load_data(path)
    assert len(data) > 0
