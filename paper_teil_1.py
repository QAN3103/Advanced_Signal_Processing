#!/usr/bin/env python
# coding: utf-8

from scipy.linalg import toeplitz
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile




def yule_walker(signal, order):
    """
    Computes the AR coefficients of a signal using the Yule-Walker equations.

    Parameters:
    signal (np.ndarray): The input signal (1D array).
    order (int): The order of the AR process (must be >= 10).

    Returns:
    tuple: (ar_coeffs, noise_variance)
        - ar_coeffs (np.ndarray): The AR coefficients (including sign).
        - noise_variance (float): The variance of the white noise.
    """

    # Berechne die Autokorrelation des Signals
    autocorr = np.correlate(signal, signal, mode='full')
    mid = len(autocorr) // 2
    r = autocorr[mid:mid + order + 1]

    R = toeplitz(r[:-1])  # Autokorrelationsmatrix
    r_vector = r[1:]     # Rechter Handvektor

    ar_coeffs = np.linalg.solve(R, r_vector)  # AR-Koeffizienten berechnen
    noise_variance = r[0] - np.dot(ar_coeffs, r_vector)  # Varianz des Rauschens

    return ar_coeffs, noise_variance


def calculate_residuals(signal, ar_coeffs):
    """
    Computes residuals.
    
    Parameters:
    signal (np.ndarray): Original Input signal
    ar_coeffs (np.ndarray): Estimated AR Coefficients of original signal
    
    Returns:
    residuals (np.ndarray)
    
    """
    
    order = len(ar_coeffs);
    residuals = np.empty(len(signal)-order)
    for i in range(order,len(signal)):
        signal2get =  signal[i-order:i]
        estimate = np.dot(ar_coeffs, signal[i-order:i][::-1])
        residuals[i-order] = signal[i] + estimate # Erste residual ist fuer i=order+1
    
    
    return residuals

def bootrsp(data, B=1):
    """
    Bootstrap resampling procedure.

    Parameters:
    data : numpy.ndarray
        Input data (1D or 2D array).
    B : int, optional
        Number of bootstrap resamples (default is 1).

    Returns:
    numpy.ndarray
        Bootstrap resamples of the input data.
        For a vector input of size [N,], returns an array of shape [N, B].
        For a matrix input of size [N, M], returns an array of shape [N, M, B].

    Example:
    out = bootrsp(np.random.randn(10, 1), 10)
    """
    if data is None:
        raise ValueError("Provide input data")

    data = np.asarray(data)
    if data.ndim > 2:
        raise ValueError("Input data can be a vector or a 2D matrix only")

    if data.ndim == 1:
        N = data.shape[0]
        resamples = np.random.choice(data, size=(N), replace=True)
    else:
        N, M = data.shape
        resamples = np.empty((N, M, B))
        for b in range(B):
            indices = np.random.randint(0, N, size=N)
            resamples[:, :, b] = data[indices, :]

    return resamples

def calculate_signal(signal, ar_coeffs, residuals):
    """
    Calculates the new signal, based on ordignal signal and residuals.
    
    Parameters:
    signal (np.ndarray): original signal
    ar_coeffs (np.ndarray): estiamted AR coefficients
    residuals (np.ndarra): residuals
    
    Returns: 
    signal_star (np.ndarray): new signal
    """
    
    order = len(ar_coeffs)
    
    N = len(signal)
    
    signal_star = np.empty(N)
    signal_star[:order] = signal[:order]  # x*(1) to x*(p)
    
    # Generate the bootstrap sample for n = p+1 to N
    for n in range(order, N):
        # Estimate the current value using AR model
        estimate = np.dot(ar_coeffs, signal_star[n-order:n][::-1])
        # Add the resampled residual to the estimate
        signal_star[n] = -(-estimate + residuals[n - order]) #TODO: Why minus?? --> Otherwise AR-coefficients have wrong sign
    
    return signal_star

def bootstrap_ar(signal, B, order):
    """
    Calculates B bootstrap estimates for AR coefficents.
    
    Parameters:
    signal (np.ndarray): orginal signal
    B (int): number of estimations
    
    Returns:
    ar_coeffs_star (matrix[B,order(10)])
    """

    ar_coeffs_orig, _ = yule_walker(signal, order)
    residuals_orig = calculate_residuals(signal, ar_coeffs_orig)
    ar_coeffs_star_all = np.zeros((B, len(ar_coeffs_orig)))
    for i in range(B):
        residual_star = bootrsp(residuals_orig)
        signal_star = calculate_signal(signal, ar_coeffs_orig,residual_star)
        # Plot generated signal, to check for stability
        if i==10 or i == 100:
            plt.figure(figsize=(10, 6))
            plt.plot(list(range(0, len(signal_star))), signal_star)
            plt.show()
                
        ar_coeffs_star,_ = yule_walker(signal_star, order)
        ar_coeffs_star_all[i] = ar_coeffs_star
    return ar_coeffs_star_all

def calculate_confidence_intervals(bootstrap_coeffs, alpha=0.05):
    lower_bound = np.percentile(bootstrap_coeffs, 100 * (alpha / 2), axis=0)
    upper_bound = np.percentile(bootstrap_coeffs, 100 * (1 - alpha / 2), axis=0)
    return lower_bound, upper_bound

def load_wave():
    samplerate, data = wavfile.read('audio/a_sound.wav')
    time = np.linspace(0, len(data) / samplerate, num=len(data))
    plt.figure(figsize=(10, 5))
    plt.plot(time, data)
    plt.title('Diskretes Zeitsignal x(n)')
    plt.xlabel('Zeit [s]')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.show()
    return data, time
if __name__ == "__main__":
    # Generate a synthetic AR(10) process
    signal, time = load_wave()

    np.random.seed(0)
    n = 1000
    true_ar_coeffs = np.array([0.8, -0.6, 0.4, -0.2, 0.1,-0.05, 0.025, -0.0125, 0.00625, -0.003125])
    order = 10
    noise = np.random.randn(n)
    #signal = np.zeros(n)

    #for i in range(10, n):
        #signal[i] = np.dot(true_ar_coeffs, signal[i-order:i][::-1]) + noise[i]
    
    # Plot generated signal
    x = list(range(0, n))
    plt.figure(figsize=(10, 6))
    plt.plot(time, signal)#
    plt.show()
    # Test yule_walker function
    estimated_ar_coeffs, noise_variance = yule_walker(signal, order)

    # Test bootstrap_ar function
    B = 1000
    bootstrap_coeffs = bootstrap_ar(signal, B, order)
    print("Bootstrap AR Coefficients (first 5):\n", bootstrap_coeffs[:5])
    lower_bound, upper_bound = calculate_confidence_intervals(bootstrap_coeffs)

    # Plot results
    plt.figure(figsize=(10, 6))

    # Go through all AR-coefficients
    for i in range(len(true_ar_coeffs)):
        plt.plot([i, i], [lower_bound[i], upper_bound[i]], color='blue', lw=2)  # Konfidenzintervall
        plt.scatter(i, true_ar_coeffs[i], color='red', zorder=5)  # Wahre AR-Koeffizienten mit Kreuz
        plt.scatter(i, bootstrap_coeffs[:, i].mean(), color='green', marker='x')  # Durchschnitt der Bootstrap-Koeffizienten

    # Plot Confidence Interval
    plt.xticks(np.arange(10), [f"AR{i+1}" for i in range(10)])
    plt.xlabel("AR-Coefficients")
    plt.ylabel("Value")
    plt.title("Confidence Interval of AR-Coefficients")
    plt.grid(True)
    plt.show()
    x = 5

