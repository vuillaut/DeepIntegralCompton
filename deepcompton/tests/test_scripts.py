# Test that scripts can actually run
from deepcompton.scripts import reconstruction_compton_density
import numpy as np
import pkg_resources

def test_reconstruction_compton_density():
    reco_filename = 'reco_theta_42_phi_104.txt'
    reconstruction_compton_density.main()
    reco_data = np.loadtxt(reco_filename)
    test_data = np.loadtxt(pkg_resources.resource_filename('deepcompton', f'data/{reco_filename}'))
    np.testing.assert_allclose(reco_data, test_data)


# def test_cone_center_distribution():
#     scripts.cone_center_distribution()