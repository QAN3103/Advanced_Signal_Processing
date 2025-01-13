#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[ ]:


def segmentation (x,l,m)
    """
    Given the data samples X=(x_1,x_2,...,x_N), the program obtains q overlapping (M<L) or non-overlapping (M>=L) segments, each of l samples 
    in the form of a matrix "y" of l rows and q columns. 
        _______________    
       |______ l ______| .....                     
       |___ m ___|______ l ______| .....         
       |___ m ___|___ m ___|______ l ______| .....

    The procedure is used for the block of blocks bootstrap.
    
    Parameters:
        x (ndarray): input vector data 
        l (int): number of elements in a segment
        m (int): shift size (i.e. L-M is the size of overlap)               

    Returns:
        y (ndarray): the output matrix of the data
        q (int): number of output segments

    Created by A. M. Zoubir and  D. R. Iskander May 1998
    
    References:
        Zhang, Y. et. al. Bootstrapping Techniques in the Estimation of Higher Order Cumulants from Short Data Records. 
        Proceedings of the International Conference on  Acoustics,  Speech  and  Signal Processing, ICASSP-93, Vol. IV, pp. 200-203.

        Zoubir, A.M. Bootstrap: Theory and Applications. 
        Proceedings of the SPIE 1993 Conference on Advanced Signal Processing Algorithms, Architectures and Implementations. pp. 216-235, San Diego, July  1993.

        Zoubir, A.M. and Boashash, B. The Bootstrap and Its Application in Signal Processing. IEEE Signal Processing Magazine, Vol. 15, No. 1, pp. 55-76, 1998.
    """
    # Flatten `x` into a 1D array.
    x = np.ravel (x)
    
    # Get the number of elements in `x`.
    n = len (x)
    
    # Calculate the number of segments.
    q = int((n - l) // m + 1)
    
    # Create an array of size [l, q].
    y = np.zeros ((l,q))
    
    # Loop through each segment and extract a slice from 'x'
    for ii in range (q):
        y[:, ii] = x [ii*m : ii*m+l]
    return y, q


# In[3]:


def segmcirc(x,l,m,q):
    """
    Given the data samples x=(x_1,x_2,...,x_N) the program obtains q overlapping (M<L) or non-overlapping (M>=L) segments, each of l samples 
    in the form of a matrix "y" of l rows and q columns. The data x_i is "wrapped" around in a circle, that is,  
    define (for i>N) x_i=x_iN, where iN=i(mod N).  
             _______________     
       .....|_______L_______| .....
       .....|____M____|_______L_______| .....
       .....|___ M ___|___ M ___|______ L ______| .....       

    The procedure is used for the circular block bootstrap.
    
    Parameters:    
        x (ndarray): input vector data 
        l (int): number of elements in a segment
        m (int): shift size (i.e. L-M is the size of overlap) 
        q (int) number of desired segments

    Returns:
        y (ndarray): the output matrix of the data

    Created by A. M. Zoubir and  D. R. Iskander May 1998

    References:
        Politis, N.P. and Romano, J.P. Bootstrap Confidence Bands for Spectra and Cross-Spectra. 
        IEEE Transactions on  Signal  Processing, Vol. 40, No. 5, 1992. 

        Zhang, Y. et. al. Bootstrapping Techniques in the Estimation of Higher Order Cumulants from Short Data Records. 
        Proceedings of the International Conference on  Acoustics, Speech and Signal Processing, ICASSP-93, Vol. IV, pp. 200-203.

        Zoubir, A.M. Bootstrap: Theory and Applications. Proceedings of the SPIE 1993 Conference on Advanced Signal 
        Processing Algorithms, Architectures and Implementations. pp. 216-235, San Diego, July 1993.

        Zoubir, A.M. and Boashash, B. The Bootstrap and Its Application in Signal Processing. 
        IEEE Signal Processing Magazine, Vol. 15, No. 1, pp. 55-76, 1998.
    """
    
   # Flatten `x` into a 1D array.
    x = np.ravel (x)
    
    # Get the number of elements in `x`.
    n = len (x)
    
    # Create an array of size [l, q].
    y = np.zeros ((l,q))
    
    # Create an array of size [q * m + l - 1, 1]
    ny = q * m + l - 1
    y_neu = np.zeros ((ny, 1))

    # Fill y_neu with values from x based on the modular operation
    r = 0
    for ii in range (ny):
        y_neu [ii] = x [ii % N]
        if ii % N == r+1:
            r = r+1
    for ii in range (q):
        y [:, ii] = y_neu [ii*m+1:ii*m+l]

