from deepcompton import utils


def test_get_test_data_path():
    path = utils.get_data_path()
    assert path.endswith('theta_42_phi_104.npy')