import sys
import numpy as np
import numpy.random as rand

sys.path.append('..')
from messages import message
sys.path.append('/Users/austin/Utilities/Code/Tensorflow')
from utils_dirichlet_tf import dirichlet_kl_divergence_tf
from utils_dirichlet_tf import dirichlet_sample_const_tf
from utils_dirichlet_tf import dirichlet_sample_tf

rand.seed(1993)

def test_sample():
    print('###############################')
    print('# Testing dirichlet_sample_tf #')
    print('###############################')
    print(dirichlet_sample_tf.__doc__)
    N1 = 100000
    N2 = 100000

    alpha = np.abs(rand.randn(5))
    beta = np.abs(rand.randn(5))
    alpha1 = np.ones((N1,5))*alpha
    beta1 = np.ones((N1,5))*beta
    X_alpha = np.vstack((alpha1,beta1))

    samples_alpha = rand.dirichlet(alpha,size=N2)
    samples_beta = rand.dirichlet(beta,size=N2)
    mean1_np = np.mean(samples_alpha,axis=0)
    mean2_np = np.mean(samples_beta,axis=0)
    std1_np = np.std(samples_alpha,axis=0)
    std2_np = np.std(samples_beta,axis=0)

    X = dirichlet_sample_tf(X_alpha)
    Xn = X.numpy()
    mean1_tf = np.mean(Xn[:N1],axis=0)
    mean2_tf = np.mean(Xn[N1:],axis=0)
    std1_tf = np.std(Xn[:N1],axis=0)
    std2_tf = np.std(Xn[N1:],axis=0)

    ts1m = np.sum(np.abs(mean1_np-mean1_tf))
    ts2m = np.sum(np.abs(mean2_np-mean2_tf))
    ts1s = np.sum(np.abs(std1_np-std1_tf))
    ts2s = np.sum(np.abs(std2_np-std2_tf))
    message(ts1m,1e-2,'mean alpha check')
    message(ts2m,1e-2,'mean beta check')
    message(ts1s,1e-2,'std alpha check')
    message(ts2s,1e-2,'std beta check')

def test_sample_const():
    print('#####################################')
    print('# Testing dirichlet_sample_const_tf #')
    print('#####################################')
    print(dirichlet_sample_const_tf.__doc__)

    alpha = np.abs(rand.randn(10))

    samples_np = rand.dirichlet(alpha,size=100000)
    mean_np = np.mean(samples_np,axis=0)
    samples_tf = dirichlet_sample_const_tf(alpha,100000)
    mean_tf = np.mean(samples_tf.numpy(),axis=0) 
    test_stat = np.sum(np.abs(mean_np-mean_tf))
    message(test_stat,1e-2,'10d mean check')

    std_np = np.std(samples_np,axis=0)
    std_tf = np.std(samples_tf.numpy(),axis=0) 
    test_stat = np.sum(np.abs(std_np-std_tf))
    message(test_stat,1e-2,'10d std check')

def test_kl_divergence():
    print('#########################################')
    print('# Testing dirichichlet_kl_divergence_tf #')
    print('#########################################')
    print(dirichlet_kl_divergence_tf.__doc__)

    beta = np.ones(5)
    alpha = 2*np.ones(5)
    kl_div = dirichlet_kl_divergence_tf(beta,beta)
    message(kl_div.numpy(),1e-7,'5d same distn check')

    kl_div = dirichlet_kl_divergence_tf(alpha,beta)
    message(np.abs(kl_div.numpy()-0.4789),1e-3,'5d different distn check')

    alpha = np.abs(rand.randn(5))
    beta = np.abs(rand.randn(5))
    kl_div = dirichlet_kl_divergence_tf(alpha,beta)
    message(np.abs(kl_div.numpy()-5.9261),1e-3,'5d random check')

if __name__ == "__main__":
    test_kl_divergence()
    test_sample_const()
    test_sample()
