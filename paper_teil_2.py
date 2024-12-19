#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np

def create_noise (sigma, p):
    """
    Creates a white Gaussian noise process with zero mean and variance sigma^2 and length p
    
    Parameters:
        sigma: variance of the noise process
       
    Returns:
        z: the white Gaussian noise 
    """
    z = np.random.normal(0, sigma, p)
    return z
     

def create_signal (a, sigma, p):
    """
    Creates a discrete-time speech signal model
    
    Parameters:
        a: AR coefficients
       
    Returns:
        x: a discrete-time speech signal model
    """
    z = create_noise (sigma, p) #generate white noise
    N = len (z) #length of output signal
    p = len (a) #model order
    x = np.zeros (N) #generate output signal
    for i in range (0, N):
        sum_a = 0
        for k in range (1, p+1):
            if i-k >= 0:  # Ensure indices are valid
                sum_a += a[k-1]*x[i-k]
        x[i]= z[i]-sum_a
    return x
        
        
    


# In[6]:


def calculate_spectrum (sigma, a, num_points):
    """
    Calculate the parametric spectrum of the speech signal 
    
    Parameters:
        a: AR coefficients
       
    Returns:
        c_xx: the parametric spectrum of the model
    """
    omega = np.linspace(-np.pi, np.pi, num_points)
    sum_a = 0
    for i in range (1,len(a)+1):
        sum_a += a[i-1]*np.exp(-1j*omega*(i-1))
        
    c_xx = sigma**2/((abs(1+sum_a))**2)
    return c_xx


# In[8]:


get_ipython().system("jupyter nbconvert --to script '/Seminar Advanced Signal Processing/grundlage.ipynb'")


# In[ ]:





# In[ ]:




