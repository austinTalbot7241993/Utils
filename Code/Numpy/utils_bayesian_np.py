import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

def extract_posterior_summary(array,post_opts):
    if post_opts['summary'] == 'median':
        med = np.median(array,axis=0)
        if 'credible_interval' in post_opts:
            if post_opts['credible_interval']:
                if 'alpha' in post_opts:
                    lb = np.quantile(array,post_opts['alpha'],axis=0)
                    ub = np.quantile(array,1-post_opts['alpha'],axis=0)
                else:
                    lb = np.quantile(array,.025,axis=0)
                    ub = np.quantile(array,.975,axis=0)
                return med,lb,ub    
        else:
            return med
    elif post_opts['summary'] == 'mean':
        mean = np.mean(array,axis=0)
        if 'credible_interval' in post_opts:
            if post_opts['credible_interval']:
                std = np.std(array,axis=0)
                if 'alpha' in post_opts:
                    z_score = st.norm.ppf(post_opts['alpha'])
                else:
                    z_score = st.norm.ppf(0.025)
                lb = mean + z_score*std
                ub = mean - z_score*std
                return med,lb,ub    
        else:
            return mean
    else: 
        raise ValueError('Not recognized')
