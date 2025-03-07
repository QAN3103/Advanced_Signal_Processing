#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[69]:

def bootstrap_statistic(x, statfun, B, *args):
    """
    Compute bootstrap statistics manually using NumPy.

    Parameters:
        x (array-like): The input time series or dataset.
        statfun (callable): Function to compute the statistic.
        B (int): Number of bootstrap resamples.
        *args: Additional arguments for the statistic function.

    Returns:
        stats (ndarray): Statistics computed for each bootstrap sample.
    """
    n = len (x)
    #create array to save statistics
    stats = np.zeros(B)
    #create matrix to store bootstrap resamples
    bootstrap_sample = np.zeros((B, n))
    #create matrix to store bootstrap indices
    bootstrap_index = np.zeros((B, n), dtype=int)  # Store indices as integers
    
    for i in range(B):
        # Generate random indices for the bootstrap sample
        index = np.random.choice(n, size=n, replace=True)
        
        # Store the bootstrap indices
        bootstrap_index[i, :] = index
        
        # Create the bootstrap sample based on the indices
        bootstrap_sample[i, :] = x[index]
        
        # Compute the statistic
        stats[i] = statfun(bootstrap_sample[i, :], *args)

    return stats, bootstrap_sample, bootstrap_index
# In[66]:


def bootstrap_univariate (input_data, B=1):
    """
    Bootstrap  resampling  procedure (univariate). 
    
    For a vector input data of size [N,1], the resampling procedure produces a matrix of size [N,B] 
    with columns being resamples of the input vector.

    For a matrix input data of size  [N,M], the resampling procedure produces a 3D matrix of  size  [N,M,B] 
    with out(:,:,i), i = 1,...,B, being a resample of the input matrix.
        
    Parameters:
        input_data - input data 
        B - number of bootstrap resamples (default B=1)  

    Returns:
        output - B bootstrap resamples of the input data  

    Created by A. M. Zoubir and D. R. Iskander May 1998
    
    References:
    Efron, B.and Tibshirani, R.  An Introduction to the Bootstrap. Chapman and Hall, 1993.

    Zoubir, A.M. Bootstrap: Theory and Applications. Proceedings of the SPIE 1993 Conference on Advanced Signal 
    Processing Algorithms, Architectures and Implementations. pp. 216-235, San Diego, July  1993.

    Zoubir, A.M. and Boashash, B. The Bootstrap and Its Application in Signal Processing. IEEE Signal Processing Magazine, 
    Vol. 15, No. 1, pp. 55-76, 1998.
    """
    #Check input shape. The function only accepts vector or 2D matrix
    s = np.shape (input_data)
    if len(s) > 2:
        raise ValueError ('Input data can be a vector or a 2D matrix only')
    else: 
    #If the input data is a vector, a matrix of size [NxB] containing the bootstrap resamples will be returned
        if min(s) == 1:
            N = len(input_data)
            output = np.zeros((N, B))
            for b in range(B):
                indices = np.random.randint(0, N, N)
                output[:, b] = input_data[indices]   
        else:
        #If the input data is a 2D matrix, a matrix of size [NxMxB] containing the bootstrap resamples will be returned
            N = s[0]
            M = s[1]
            output = np.zeros((N, M, B))
            for b in range(B):
                indices = np.random.randint(0, N * M, N * M)
                row_indices = indices // M
                col_indices = indices % M
                output[:, :, b] = input_data[row_indices.reshape(N, M), col_indices.reshape(N, M)]
        return output


# In[65]:


def bootstrap_bivariate (input_1, input_2, B=1):
    """
    Bootstrap  resampling  procedure for bivariate data. For a vector input data of size [N,1], the  resampling procedure produces a matrix of size [N,B] with columns 
    being resamples of the input vector.

    Parameters:
        input_1 - input data (first variate)
        input_2 - input data (second variate). If input_2 is not provided, the function runs bootrap_univariate by default.
        B - number of bootstrap resamples (default B=1) 
    
    Returns: 
        output_1 - B bootstrap resamples of the first variate
        output_2 - B bootstrap resamples of the second variate

    Created by A. M. Zoubir and D. R. Iskander May 1998
    
    References:
        Efron, B.and Tibshirani, R.  An Introduction to the Bootstrap. Chapman and Hall, 1993.
    
        Zoubir, A.M. Bootstrap: Theory and Applications. Proceedings of the SPIE 1993 Conference on Advanced Signal 
        Processing Algorithms, Architectures and Implementations. pp. 216-235, San Diego, July  1993.
        
        Zoubir, A.M. and Boashash, B. The Bootstrap and Its Application in Signal Processing. IEEE Signal Processing Magazine, Vol. 15, No. 1, pp. 55-76, 1998.
    """

    s1 = np.shape (input_1)
    s2 = np.shape (input_2)
    if len(s1) > 2 or len (s2) > 2:
        raise ValueError ('Input data can be vectors or a 2D matrices only')
    else: 
        if s1 != s2:
            raise ValueError("The input vectors must have the same size.")
    #if s1==fliplr(s2),
    #  in2=in2.';
    #end;
        else: 
            if min(s1)==1: 
                N = len(s1)
                output = np.zeros((N, B))
                for b in range(B):
                    indices = np.random.randint(0, N, N)
                    output_1 [:, b] = input_1[indices]
                    output_2 [:, b] = input_2[indices]
            else:
                N = s1[0]
                M = s1[1]
                output = np.zeros((N, M, B))
                for b in range(B):
                    indices = np.random.randint(0, N * M, N * M)
                    row_indices = indices // M
                    col_indices = indices % M
                    output_1[:, :, b] = input_1[row_indices.reshape(N, M), col_indices.reshape(N, M)]
                    output_2[:, :, b] = input_2[row_indices.reshape(N, M), col_indices.reshape(N, M)]
                    
            return output_1, output_2






