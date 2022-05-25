import sys
import numpy as np
import numpy.random as rand

sys.path.append('../../Code/Miscellaneous')
from utils_unitTest import tolerance,greater_than,message
from utils_unitTest import print_otm,print_mtm,print_ftm
from utils_unitTest import time_method

sys.path.append('/Users/austin/Utilities/Code/Numpy')
from utils_dirichlet_np import dirichlet_kl_divergence_np

rand.seed(1993)

def test_dirichlet_kl_divergence_np():
    print_mtm('dirichlet_kl_divergence_np')
    beta = np.ones(5)
    alpha = 2*np.ones(5)
    kl_div = dirichlet_kl_divergence_np(beta,beta)
    message(kl_div,1e-7,'5d same distn check')

    kl_div = dirichlet_kl_divergence_np(alpha,beta)
    message(np.abs(kl_div-0.4789),1e-3,'5d different distn check')

    alpha = np.abs(rand.randn(5))
    #[0.17720752 0.25235584 0.20237865 0.51730492 0.51398868]
    beta = np.abs(rand.randn(5))
    #[0.4579376  1.7820728  0.05645113 1.28889974 0.28956598]
    kl_div = dirichlet_kl_divergence_np(alpha,beta)
    message(np.abs(kl_div-5.9261),1e-3,'5d random check')

if __name__ == "__main__":
    print_ftm('utils_dirichlet_np')
    test_dirichlet_kl_divergence_np()

