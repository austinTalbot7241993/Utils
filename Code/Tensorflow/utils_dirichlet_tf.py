import tensorflow as tf

def dirichlet_sample_tf(alpha):
    '''
    This method generates samples from multiple Dirichlet distributions.
    Useful when implementing variational autoencoders
    distribution

    Parameters
    ----------
    alpha : array-like(n_samples,K)
        Parameters for distributions 

    Returns
    -------
    samples_dirichlet : tf.Float, array-like(n_samples,K)
        The samples
    '''
    k = alpha.shape[1]
    samples_gamma = [tf.random.gamma([1],alpha[:,i],
                                                    1) for i in range(k)]
    samples_gmat = tf.stack(samples_gamma)
    sum_k = tf.reduce_sum(samples_gmat,axis=0)
    samples_dirichlet = tf.transpose(samples_gmat/sum_k)
    return samples_dirichlet

def dirichlet_sample_const_tf(alpha,n_samples):
    '''
    This method generates multiple samples from the same Dirichlet 
    distribution

    Parameters
    ----------
    alpha : array-like(K,)
        Parameters for distribution 

    n_samples : int
        Number of samples

    Returns
    -------
    samples_dirichlet : tf.Float, array-like(n_samples,K)
        The samples
    '''
    k = len(alpha)
    samples_gamma = [tf.random.gamma([n_samples],alpha[i],
                                                    1) for i in range(k)]
    samples_gmat = tf.stack(samples_gamma)
    sum_k = tf.reduce_sum(samples_gmat,axis=0)
    samples_dirichlet = tf.transpose(samples_gmat/sum_k)
    return samples_dirichlet
    

def dirichlet_kl_divergence_tf(alpha,beta):
    '''
    This method computes the KL divergence between two dirichlet 
    distributions in Tensorflow 

    Parameters
    ----------
    alpha : array-like(K,)
        Parameters for first distribution 

    beta : array-like(K,)
        Parameters for second distribution 

    Returns
    -------
    kl_divergence : tf.Float
        The KL divergence
    '''
    alpha_0 = tf.reduce_sum(alpha)
    beta_0 = tf.reduce_sum(beta)
    t1 = tf.math.lgamma(alpha_0)
    t2 = -1*tf.reduce_sum(tf.math.lgamma(alpha))
    t3 = -1*tf.math.lgamma(beta_0)
    t4 = tf.reduce_sum(tf.math.lgamma(beta))
    t5 = tf.reduce_sum((alpha - beta)*(tf.math.digamma(alpha) - 
                                                tf.math.digamma(alpha_0)))
    return t1 + t2 + t3 + t4 + t5

def dirichlet_entropy_tf(alpha):
    alpha_0 = tf.reduce_sum(alpha)
    log_B_alpha_num = np.sum([tf.math.lgamma(alpha[i]) for i in range(K)])
    log_B_alpha_denom = tf.math.lgamma(alpha_0)

    term1 = log_B_alpha_num - log_B_alpha_denom
    term2 = (alpha_0-K)*tf.math.digamma(alpha_0)
    dga = tf.math.digamma(alpha)
    term3 = tf.reduce_sum((alpha-1)*dga)

    entropy = term1 + term2 + term3
    return entropy

