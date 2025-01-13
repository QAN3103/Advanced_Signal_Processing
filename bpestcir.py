#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import segments as seg 
import bootstrap_resampling as bstr


# In[ ]:


def bpestcir(x,estfun,l1,m1,q1,l2,m2,q2,B=99,*args)
    """
    The program calculates the estimate and the variance of an estimator of a parameter from the input vector X.
    The algorithm is based  on a circular block bootstrap and is suitable when the data is weakly correlated. 
    
    Parameters:
        x (ndarray):  input vector data 
        estfun (callable): the estimator of the parameter 
        l1 (int): number of elements in the first block (see "segmcirc.m")
        m1 (int): shift size in the first block
        q1 (int): number of segments in the first block
        l2 (int): number of elements in the second block (see "segmcirc.m")
        m2 (int): shift size in the second block
        q2 (int): number of segments in the second block
        B (int): number of bootstrap resamplings (default B=99)
        PAR1,... - other parameters than x to be passed to estfun

    Returns:  
        est - estimate of the parameter
        estvar - variance of the estimator  

    Created by A. M. Zoubir and D. R. Iskander May 1998
    
    References:
        Politis, N.P. and Romano, J.P. Bootstrap Confidence Bands for  Spectra and Cross-Spectra. 
        IEEE Transactions on  Signal  Processing, Vol. 40, No. 5, 1992. 

        Zhang, Y. et. al. Bootstrapping Techniques in the Estimation of Higher Order Cumulants from Short Data Records. 
        Proceedings of the International Conference on  Acoustics, Speech and Signal Processing, ICASSP-93, Vol. IV, pp. 200-203.

        Zoubir, A.M. Bootstrap: Theory and Applications. 
        Proceedings of the SPIE 1993 Conference on Advanced Signal Processing Algorithms, Architectures and Implementations. pp. 216-235, San Diego, July  1993.

        Zoubir, A.M. and Boashash, B. The Bootstrap and Its Application in Signal Processing. 
        IEEE Signal Processing Magazine, Vol. 15, No. 1, pp. 55-76, 1998.

    """
    # ensure data is 1D array
    x = np.ravel (x)
    
    # Generate first block of circular segments
    ql = seg.segmcirc (x, l1, m1, q1)
    
    # Step 2: Apply the estimator function to the segments
    estm = estfun(ql, *args)
    
    # Generate second block of circular segments
    beta = seg.segmcirc (estm, l2, m2, q2)
    
    # Bootstrap resampling
    ind = bstr.bootstrap_univariate (1:q2, B)
    y = beta [ind]
    
    # Calculate the mean over bootstrap resamples
    estsm = np.mean (y)
    
    # Final estimate and variance
    est = np.mean (estsm)
    estvar = np.var (estsm)
    return est, estvar
    


