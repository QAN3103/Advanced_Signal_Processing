#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[4]:


def correl(input_1,input_2):
    """
    Calculate the correlation coeficientof two vectors - input_1 and input_2 or gives the vector of correlation coeficients
    of the raws of matrix input1 and input2. Note, that input_1 must have the same size as input_2.
    
    Parameters:
        input_1 (array-like): The first input array or matrix. Can be a 1D vector or a 2D matrix.
        input_2 (array-like): The second input array or matrix. Must have the same size as `input_1`.
    
    Returns:
        cor (float or ndarray): correlation coefficient of input_1 and input_2
    """
    s1 = np.shape (input_1)
    s2 = np.shape (input_2)
    if s1 == s2:
        if s1 [0] == 1 or s1[1] == 1:
            input_1 = np.ravel(input_1)  # Flatten to 1D array
            input_2 = np.ravel(input_2)
        mx1 = np.mean (input_1)
        mx2 = np.mean (np.power(input_1, 2))
        my1 = np.mean (input_2)
        my2 = np.mean (np.power(input_2, 2))
        Sxx = mx2 - np.power(mx1, 2)
        Syy = my2 - np.power(my1, 2)
        Sxy = np.mean (input_1 * input_2) - mx1 * my1
        Rxy = Sxy / np.sqrt (Sxx * Syy)
        return Rxy
    else:
        raise ValueError("The input vectors must have the same size.")


# In[ ]:




