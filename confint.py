#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import bootstrap_resampling as bstr


# In[2]:


def confint(x,statfun,alpha = 0.05, b1 = 199, b2 = 25, *args):
    """
    Calculate the 100(1-alpha)% confidence interval of the estimator of a parameter based on the bootstrap percentile-t method  

    Parameters: 
        x (array-like): input vector data 
        statfun (callable): the estimator of the parameter   
        alpha (float): level of significance, must be non-negative (default alpha=0.05)  
        b1 (int): number of bootstrap resamplings (default B1=199)
        b2 (int): number of bootstrap resamplings for variance estimation (nested bootstrap) (default B2=25)    
        PAR1,... - other parameters than x to be passed to statfun

    Returns: 
        Lo (float): The lower bound 
        Up (float): The upper bound

    Created by A. M. Zoubir and D. R. Iskander May 1998
    
    References:
        Efron, B.and Tibshirani, R.  An Introduction to the Bootstrap. Chapman and Hall, 1993.

        Hall, P. Theoretical Comparison of Bootstrap Confidence Intervals. 
        The Annals of Statistics, Vol  16, No. 3, pp. 927-953, 1988.

        Zoubir, A.M. Bootstrap: Theory and Applications. 
        Proceedings of the SPIE 1993 Conference on Advanced Signal Processing Algorithms, Architectures and Implementations. pp. 216-235, San Diego, July  1993.

        Zoubir, A.M. and Boashash, B. The Bootstrap and Its Application in Signal Processing. IEEE Signal Processing Magazine, 
        Vol. 15, No. 1, pp. 55-76, 1998.

    """
    #flatten input data
    x = np.ravel(x)
    
    #compute the estimator
    vhat = statfun (x, *args)

    # create b1 bootstrap 
    vhatstar, b1_resample, index = bstr.bootstrap_statistic(x, statfun, b1, *args)
    
    #perform b2 nested bootstrap from the statistics of b1 bootstrap resamples
    b2_nested_stat = bstr.bootstrap_statistic(x[index], statfun, b2, *args) [0]
    
    #create b2 independent bootstrap resamples
    b2_stat = bstr.bootstrap_statistic(x, statfun, b2, *args) [0]
    
    # compute variance of the b2 bootstrap resamples
    sigma = np.std (b2_stat)
    
    #compute variance of the nested bootstrap
    sigma_nested = np.std (b2_nested_stat)
    
    #compute index for the confidence interval 
    q1 = round(b1 * alpha * 0.5)   
    q2 = b1-q1+1
    
    #calculate pivotal statistic
    tvec=(vhatstar-vhat)/sigma_nested; 
    
    #sort tvec in ascending order and save the corresponding index
    st = np.sort (tvec)
    ind = np.argsort (tvec)
    
    lo = vhat - st [q1]*sigma
    up = vhat - st [q2]*sigma
    
    return lo, up


# In[ ]:


def confinh(x,statfun,alpha,b,*args):
    """
    Calculate the 100(1-alpha)% confidence interval of the estimator of a parameter based on the bootstrap hybrid method  
    
    Parameters:
        x (array): input vector data 
        statfun (callable): the estimator of the parameter
        alpha (float): level of significance (default alpha=0.05)  
        b (int): number of bootstrap resamplings (default B1=199) 
        PAR1,... - other parameters than x to be passed to statfun
        
    Returns:
        lo - The lower bound 
        up - The upper bound

    Created by A. M. Zoubir and D. R. Iskander. May 1998

    References:
        Efron, B.and Tibshirani, R.  An Introduction to the Bootstrap. Chapman and Hall, 1993.

        Hall, P. Theoretical Comparison of Bootstrap Confidence Intervals. 
        The Annals of Statistics, Vol  16, No. 3, pp. 927-953, 1988.

        Zoubir, A.M. Bootstrap: Theory and Applications. 
        Proceedings of the SPIE 1993 Conference on Advanced  Signal Processing Algorithms, Architectures and Implementations. pp. 216-235, San Diego, July  1993.

        Zoubir, A.M. and Boashash, B. 
        The Bootstrap and Its Application in Signal Processing. IEEE Signal Processing Magazine, Vol. 15, No. 1, pp. 55-76, 1998.
    """
    #flatten input data
    x = np.ravel (x)
    
    #calculate the statistic
    vhat = statfun (x, *args)
    
    #perform bootstrap and calculate the bootstrap statistics
    vhatstar = bstr.bootstrap_statistic (x, statfun, b, *args)
    
    #compute tvec
    tvec=vhatstar-vhat;
    
    #compute the index to build the confidence interval
    q1=round(b*alpha*0.5)
    q2=b-q1+1;
    
    #sort the computed bootstrap statistics in ascending order
    st = np.sort(tvec)
    
    #compute the confidence interval
    lo = vhat-st(q2)
    up = vhat-st(q1);

    return lo, up


# In[4]:


def confintp(x,statfun,alpha,b,*args): 
    """
    Calculate the 100(1-alpha)% confidence interval of the estimator of a parameter based on the bootstrap percentile method  

    Inputs:
        x (array): input vector data 
        statfun (callable): the estimator of the parameter   
        alpha (float): level of significance (default alpha=0.05)  
        b (int): number of bootstrap resamplings (default B1=199), must be non-negative   
        PAR1,... - other parameters than x to be passed to statfun
        
    Returns:
        lo (float): The lower bound 
        Up (float): The upper bound

    Created by A. M. Zoubir and  D. R. Iskander. May 1998

    References:
        Efron, B.and Tibshirani, R.  An Introduction to the Bootstrap. Chapman and Hall, 1993.

        Hall, P. Theoretical Comparison of Bootstrap Confidence Intervals. The Annals of Statistics, Vol  16, No. 3, pp. 927-953, 1988.

        Zoubir, A.M. Bootstrap: Theory and Applications. 
        Proceedings of the SPIE 1993 Conference on Advanced  Signal Processing Algorithms, Architectures and Implementations. pp. 216-235, San Diego, July  1993.

        Zoubir, A.M. and Boashash, B. 
        The Bootstrap and Its Application in Signal Processing. IEEE Signal Processing Magazine, Vol. 15, No. 1, pp. 55-76, 1998.
    """
    #flatten input data
    x = np.ravel(x)
    
    #calculate statistic
    vhat = statfun (x, *args)
    
    #perform bootstrap and calculate bootstrap statistics
    vhatstar = bstr.bootstrap_statistic (x, statfun, b, *args) [0]
    
    #calculate index of the estimated parameter
    q1 = round(b*alpha*0.5)
    q2 = b-q1+1
    
    #sort the calculated bootstrap statistics in ascending order
    st = np.sort(vhatstar)
    
    lo = st[q1]
    up = st[q2]
    
    return lo, up



# In[ ]:




