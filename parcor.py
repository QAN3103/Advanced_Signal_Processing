
import numpy as np
import correlation as corr
import bootstrap_resampling as bstr


def compute_parcor(signal, lag):
    """
    Compute the partial autocorrelation (PARCOR) coefficients using the Levinson-Durbin recursion.

    Parameters:
        signal (np.ndarray): The input time series.
        lag (int): The maximum lag to compute PARCOR coefficients.

    Returns:
        parcor (np.ndarray): The PARCOR coefficients of length `lag`.
    """
    # Ensure signal is a numpy array
    signal = np.asarray(signal)
    n = len(signal)
    if n <= lag:
        raise ValueError("Signal length must be greater than the lag.")
    
    # Autocorrelation calculation
    r = np.correlate(signal, signal, mode="full")
    r = r[len(signal)-1:]  # Use only the positive lags

    # Initialize Levinson-Durbin recursion
    a = np.zeros(lag + 1)
    e = r[0]
    parcor = np.zeros(lag)

    for k in range(1, lag + 1):
        # Compute reflection coefficient
        lambda_k = (r[k] - np.dot(a[1:k][::-1], r[1:k])) / e
        parcor[k-1] = lambda_k

        # Update coefficients
        a[1:k+1] = a[1:k+1] + lambda_k * a[k-1::-1]
        e *= (1 - lambda_k ** 2)

    return parcor

def parcor_bootstrap(signal, lag, B):
    """
    Compute bootstrap PARCOR coefficients for a given signal.

    Parameters:
        signal (np.ndarray): The input time series.
        lag (int): The maximum lag to compute PARCOR coefficients.
        B (int): The number of bootstrap resamples.

    Returns:
        bootstrap_pacf (np.ndarray): Bootstrap PARCOR coefficients of shape [B, lag].
    """
    # Ensure signal is a numpy array
    signal = np.asarray(signal)
    if signal.ndim != 1:
        raise ValueError("Signal must be a 1D array.")
    
    # Generate bootstrap samples
    #bootstrap_samples = np.array([np.random.choice(signal, size=len(signal), replace=True) for _ in range(B)])
    bootstrap_samples = bstr.bootstrap_statistic (signal, np.mean, B) [1]
    
    # Initialize storage for PARCOR coefficients
    bootstrap_pacf = np.zeros((B, lag))

    # Compute PARCOR for each bootstrap sample
    for b in range(B):
        bootstrap_pacf[b, :] = compute_parcor(bootstrap_samples[b], lag)
    
    return bootstrap_pacf

def parcor_to_ar(parcor):
    """
    Compute AR coefficients from PARCOR (reflection) coefficients.

    Parameters:
        parcor (np.ndarray): The PARCOR coefficients of length `p` (order of the AR model).

    Returns:
        ar_coeffs (np.ndarray): The AR coefficients of length `p` (excluding the intercept term).
    """
    # Ensure PARCOR is a numpy array
    parcor = np.asarray(parcor)
    p = len(parcor)

    # Initialize AR coefficients
    ar_coeffs = np.zeros(p)

    for k in range(p):
        # Add the new PARCOR coefficient
        ar_coeffs[k] = parcor[k]

        # Update AR coefficients for previous lags
        for j in range(k):
            ar_coeffs[j] = ar_coeffs[j] - parcor[k] * ar_coeffs[k - j - 1]

    return ar_coeffs

def ar_to_parcor(ar_coeffs):
    """
    Compute PARCOR (partial autocorrelation) coefficients from AR coefficients.

    Parameters:
        ar_coeffs (np.ndarray): Autoregressive (AR) coefficients of shape (p,).

    Returns:
        parcor (np.ndarray): PARCOR coefficients of shape (p,).
    """
    # Ensure ar_coeffs is a numpy array
    ar_coeffs = np.asarray(ar_coeffs)
    p = len(ar_coeffs)
    
    # Initialize PARCOR coefficients
    parcor = np.zeros(p)
    
    # Start with the AR coefficients and compute backward recursion
    ar_current = ar_coeffs.copy()
    
    for k in range(p, 0, -1):
        # Last AR coefficient is the PARCOR coefficient for this lag
        parcor[k-1] = ar_current[-1]
        
        # Update AR coefficients using the Levinson-Durbin recursion
        if k > 1:
            ar_current = (
                ar_current[:-1] - parcor[k-1] * ar_current[-2::-1]
            )
    
    return parcor

def bootstrap_parcor_similarity(x, x_noise, ar_order, B=1000):
    """
    This function computes the similarity between the dynamics of two signals by comparing the bootstrap distributions of their PARCOR coefficients. 
    The similarity is based on the overlap of the probability density functions (PDFs) of PARCOR coefficients for the two signals
    Parameters:
        x (array-like): The original clean signal
        x_noise (array-like): The noisy signal
        ar_order (int): The order of the autoregressive (AR) model used to compute the PARCOR coefficients.
        B (int) : The number of bootstrap samples to generate for estimating the distribution of PARCOR coefficients (default=1000)
    
    Returns:
        similarity (float): A scalar value between 0 and 1 representing the similarity between the dynamics of the clean and noisy signals.
            - 1: The distributions are identical (perfect similarity).
            - 0: The distributions are completely disjoint (no similarity).
    """
    pacf = par.parcor_bootstrap (x, ar_order, B)
    pacf_x_noise = par.parcor_bootstrap (x_noise, ar_order, B)
    # Create empty list to store overlap area for each PARCOR coefficient
    overlap_area = []
    # Iterate over each PARCOR coefficient up to the AR order  
    for k in range(ar_order):
        # estimate the probability density function (PDF) of the PARCOR of the true signal using Gaussian kernel
        kde_x = stats.gaussian_kde(pacf[:, k])
        # estimate the probability density function (PDF) of the PARCOR of the noisy signal using Gaussian kernel
        kde_x_noise = stats.gaussian_kde(pacf_x_noise[:, k])
        # # Define the range for integration based on the minimum and maximum values of the KDE datasets
        r_range = np.linspace(min(kde_x.dataset.min(), kde_x_noise.dataset.min()),
                              max(kde_x.dataset.max(), kde_x_noise.dataset.max()), 1000)
        # Compute the overlap area numerically by integrating the minimum of the two PDFs
        overlap = np.trapz(np.minimum(kde_x(r_range), kde_x_noise(r_range)), r_range)
        overlap_area.append(overlap)
    
    # Compute the mean overlap area across all PARCOR coefficients as the similarity measure
    similarity = np.mean(overlap_area)
    return similarity

    
    

