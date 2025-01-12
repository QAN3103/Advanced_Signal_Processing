#!/usr/bin/env python
# coding: utf-8

from scipy.linalg import toeplitz
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt

def yule_walker(signal):
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
    order = 10;
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
    
    order = 10;
    residuals = np.empty(len(signal))
    for i in range(len(signal)-order):
        estimate = np.dot(ar_coeffs, signal[i:i + order][::-1])
        residuals[i+order] = signal[i+order] + estimate
    residuals = residuals[order:]
    
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

def calcualte_signal(signal, ar_coeffs, residuals):
    """
    Calculates the new signal, based on ordignal signal and residuals.
    
    Parameters:
    signal (np.ndarray): original signal
    ar_coeffs (np.ndarray): estiamted AR coefficients
    residuals (np.ndarra): residuals
    
    Returns: 
    signal_star (np.ndarray): new signal
    """
    
    order = 10
    
    signal_star = np.empty(len(signal))
    
    for i in range(order):
        signal_star[i] = signal[i]
    for i in range(order, len(signal)):
        estimate = np.dot(ar_coeffs, signal_star[i-order:i][::-1])
        signal_star[i] = -estimate + residuals[i-order]
        
    return signal_star

def bootstrap_ar(signal, B):
    """
    Calculates B bootstrap estimates for AR coefficents.
    
    Parameters:
    signal (np.ndarray): orginal signal
    B (int): number of estimations
    
    Returns:
    ar_coeffs_star (matrix[B,order(10)])
    """
    
    ar_coeffs_orig,_ = yule_walker(signal)
    ar_coeffs_orig_2, _ = sm.regression.yule_walker(signal, order=10,
                                       method="mle")
    residuals_orig = calculate_residuals(signal, ar_coeffs_orig)
    ar_coeffs_star_all = np.zeros((B, len(ar_coeffs_orig)))
    for i in range(B):
        residual_star = bootrsp(residuals_orig)
        signal_star = calcualte_signal(signal, ar_coeffs_orig,residual_star)
        ar_coeffs_star,_ = yule_walker(signal_star)
        ar_coeffs_star_all[i] = ar_coeffs_star
    return ar_coeffs_star_all

def calculate_confidence_intervals(bootstrap_coeffs, alpha=0.05):
    lower_bound = np.percentile(bootstrap_coeffs, 100 * (alpha / 2), axis=0)
    upper_bound = np.percentile(bootstrap_coeffs, 100 * (1 - alpha / 2), axis=0)
    return lower_bound, upper_bound
    
if __name__ == "__main__":
    # Generate a synthetic AR(10) process
    np.random.seed(0)
    n = 1000
    true_ar_coeffs = np.array([0.75, -0.5, 0.25, 0.1, 0.05, 0, 0, 0, 0, 0])
    noise = np.random.randn(n)
    signal = np.zeros(n)

    for i in range(10, n):
        signal[i] = np.dot(true_ar_coeffs, signal[i-10:i][::-1]) + noise[i]

    # Test yule_walker function
    estimated_ar_coeffs, noise_variance = yule_walker(signal)
    print("Estimated AR Coefficients:", estimated_ar_coeffs)
    print("Noise Variance:", noise_variance)

    # Test bootstrap_ar function
    B = 100
    bootstrap_coeffs = bootstrap_ar(signal, B)
    print("Bootstrap AR Coefficients (first 5):\n", bootstrap_coeffs[:5])
    lower_bound, upper_bound = calculate_confidence_intervals(bootstrap_coeffs)

    # Plotten der Ergebnisse
    plt.figure(figsize=(10, 6))

    # AR-Koeffizienten
    ar_coeffs = np.array([0.75, -0.5, 0.25, 0.1, 0.05, 0, 0, 0, 0, 0])

    # Schleife durch alle AR-Koeffizienten und zeichnen
    for i in range(len(ar_coeffs)):
        plt.plot([i, i], [lower_bound[i], upper_bound[i]], color='blue', lw=2)  # Konfidenzintervall
        plt.scatter(i, ar_coeffs[i], color='red', zorder=5)  # Wahre AR-Koeffizienten mit Kreuz
        plt.scatter(i, bootstrap_coeffs[:, i].mean(), color='green', marker='x')  # Durchschnitt der Bootstrap-Koeffizienten

    # Achsenbeschriftungen und Titel
    plt.xticks(np.arange(10), [f"AR{i+1}" for i in range(10)])
    plt.xlabel("AR-Koeffizienten")
    plt.ylabel("Wert")
    plt.title("Konfidenzintervalle fuer AR-Koeffizienten (Bootstrap)")
    plt.grid(True)
    plt.show()


