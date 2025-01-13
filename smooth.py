#!/usr/bin/env python
# coding: utf-8

# In[17]:


import numpy as np


# In[19]:


def smooth(x, y, w):
    """
    A running line smoother that fits the data by linear least squares. Used to compute the variance stabilising transformation.

    Parameters:  
        x (ndarray): one or more columns of covariates
        y (ndarray): one column of response for each column of covariate
        w (float): span, proportion of data in symmetric centred window
        
    Returns:  
        x_sort (ndarray): sorted columns of x
        y_sort (ndarray): values of y associated with x_sort
        y_sm (ndarray): smoothed version of y

    Note: If inputs are row vectors, operation is carried out row-wise.

    Created by A. M. Zoubir and Hwa-Tung Ong, 1996

    References
        Hastie, T.J. and Tibshirani, R.J. Generalised additive models. Chapman and Hall, 1990.  

        Zoubir, A.M.   Bootstrap: Theory and Applications. 
        Proceedings of the SPIE 1993 Conference on Advanced Signal Processing Algorithms, Architectures and Implementations. pp. 216-235, San Diego, July  1993.

        Zoubir, A.M. and Boashash, B. 
        The Bootstrap and Its Application in Signal Processing. IEEE Signal Processing Magazine, Vol. 15, No. 1, pp. 55-76, 1998.
    """
    s1 = np.shape(x)
    s2 = np.shape(y)
    if s1 == s2:
        (nr, nc) = np.shape(x)
        n = nr
        if (nr == 1):
            x = x.T
            y = y.T
            n = nc
            nc = 1         
        y_sm = np.zeros((n,nc))
        x_sort = np.sort(x, axis=0)
        order = np.argsort(x, axis = 0)
        y_sort = np.zeros_like(y)
        for i in range(nc):
            y_sort[:, i] = y[order[:, i], i]
        k = int(np.trunc (w*n/2))
        for i in range (n):
            window = range(max(i - k, 0), min(i + k + 1, n))
            xwin = x_sort[window,:]
            ywin = y_sort[window,:]
            xbar = np.mean(xwin)
            ybar = np.mean(ywin)
            x_mc = xwin - xbar    # mc = mean-corrected
            y_mc = ywin - ybar;
            y_sm[i,:] = np.sum(x_mc * y_mc) / np.sum(x_mc * x_mc) * (x_sort [i,:] - xbar) + ybar
        if nr==1:
            x_sort = x_sort.T
            y_sort = y_sort.T
            y_sm = y_sm.T
    else:
        raise ValueError("The input data must have the same size.")
    









