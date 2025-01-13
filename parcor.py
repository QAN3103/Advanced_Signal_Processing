
import numpy as np
import correlation as corr


def compute_parcor(x, lag):
    """
    Compute PARCOR coefficients for a single data using Levinson-Durbin recursion

    Parameters:
        x (array-like): The input time series.
        lag (int): Time lag to compute PARCOR coefficients.

    Returns:
        pacf (ndarray): PARCOR coefficients for the original data.
    """
    # Check size of input data
    n = len(x)

    # Create an array to store the autocorrelations
    r = np.zeros(lag + 1)
    
    # Calculate autocorrelations
    for k in range(lag + 1):
        r[k] = correl(x[:n - k], x[k:])

    #initiate array to store parcor
    pacf = np.zeros(lag)
    
    #initiate array to store reflection coefficients
    a = np.zeros((lag, lag))  
    
    # Compute PARCOR coefficients using Levinson-Durbin recursion
    #1st iteration - calculate 1st parcor
    a[0, 0] = r[1] / r[0]
    
    #save 1st parcor
    pacf[0] = a[0, 0]
    
    #initiate the first variance of residual signal
    sigma = r[0] * (1 - pacf[0] ** 2)
    
    #perform the next iterations
    for k in range(1, lag):
        for j in range(k):
        #calculate parcor
            a[k, k] = (r[k + 1] - sum(a[j, k - 1] * r[k - j])) / sigma
        #save parcor
        pacf[k] = a[k, k]
        #update previous parcor
        for j in range(k):
            a[j, k] = a[j, k - 1] - a[k, k] * a[k - j - 1, k - 1]
        #update variance 
        sigma2 *= (1 - pacf[k] ** 2)
    return pacf

def parcor_bootstrap (x, lag, B):
    """
    Compute PARCOR coefficients for a dataset using Levinson-Durbin recursion and perform bootstrapping.

    Parameters:
        x (array-like): The input time series.
        lag (int): Time lag to compute PARCOR coefficients.
        B (int): Number of bootstrap resamples.

    Returns:
        bootstrap_pacf (ndarray): PARCOR coefficients for bootstrap samples of shape [lag, B].
    """
    # Bootstrap resampling
    sample = bootstrap_univariate(x, B)
    
    #initiate array to store parcor of the input data
    bootstrap_pacf = np.zeros((lag, B))
    
    # Compute PARCOR coefficients for each bootstrap sample
    for b in range(B):
        bootstrap_pacf[:, b] = compute_parcor(sample[:, b], lag)
    return bootstrap_pacf
    
    

