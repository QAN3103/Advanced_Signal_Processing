#!/usr/bin/env python
# coding: utf-8

from math import pi
from scipy.linalg import toeplitz
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import resample




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
    # Step 1
    ar_coeffs_orig, _ = yule_walker(signal, order)
    residuals_orig = calculate_residuals(signal, ar_coeffs_orig)
    
    # Initialization of Step 2-4
    ar_coeffs_star_all = np.zeros((B, len(ar_coeffs_orig)))
    spectrum_all = np.zeros((B, 80+1))
    variance_all = np.zeros(B)
    for i in range(B):
        # Step 2
        residual_star = bootrsp(residuals_orig)
        signal_star = calculate_signal(signal, ar_coeffs_orig,residual_star)
        
        # Plot generated signal, to check for stability
        # if i==10 or i == 100:
        #     plt.figure(figsize=(10, 6))
        #     plt.title('Bootstrap Sample')
        #     plt.plot(list(range(0, len(signal_star))), signal_star)
        #     plt.show()
            
        # Step 3        
        ar_coeffs_star, variance_star = yule_walker(signal_star, order)
        
        # Save all Bootstrap estimates
        ar_coeffs_star_all[i] = ar_coeffs_star
        variance_all[i] = variance_star
        
        # Calculate Spectrum of each Bootstrap estimate
        spectrum = calculate_spectrum(ar_coeffs_star, variance_star)
        spectrum_all[i] = spectrum
        
    return ar_coeffs_star_all, variance_all, spectrum_all

def calculate_confidence_intervals(bootstrap_coeffs, alpha=0.05):
    lower_bound = np.percentile(bootstrap_coeffs, 100 * (alpha / 2), axis=0)
    upper_bound = np.percentile(bootstrap_coeffs, 100 * (1 - alpha / 2), axis=0)
    median = np.percentile(bootstrap_coeffs, 50, axis=0)
    return lower_bound, upper_bound, median

def calculate_spectrum(ar_coeffs, variance, N=160):
    k = np.arange(0, N // 2 + 1)  # Create an array of k from 0 to N/2
    omega = 2 * np.pi * k / N   # Calculate ωk for each k
    Cxx = np.empty(len(omega))
    for i in range(len(omega)):
        sum = 0
        for k in range(len(ar_coeffs)):
            sum = sum + ar_coeffs[k] * np.exp(-omega[i] * (k + 1) * 1j)
        Cxx[i] = variance / pow(np.abs(1+ sum), 2)
    return Cxx

def load_wave(filepath, downsample):
    samplerate, data = wavfile.read(filepath)
    time = np.linspace(0, len(data) / samplerate, num=len(data))
    plt.figure(figsize=(10, 5))
    plt.plot(time, data)
    plt.title('Original Timeseries')
    plt.xlabel('Zeit [s]')
    plt.ylabel('Amplitude')
    plt.grid()
    if downsample:
        data = resample(data,160)
        time = resample(time,160)
        plt.figure(figsize=(10, 5))
        plt.plot(np.linspace(1,160,160), data)
        plt.title('Resampled Timeseries (N=160)')
        plt.xlabel('Zeit [s]')
        plt.ylabel('Amplitude')
        plt.grid()
    else:
        N = 500
        data = resample(data,N)
        time = resample(time, N)
        plt.figure(figsize=(10, 5))
        plt.plot(np.linspace(1, N,N), data)
        plt.title('Resampled Timeseries (N=high default)')
        plt.xlabel('Zeit [s]')
        plt.ylabel('Amplitude')
        plt.grid()
    plt.show()

    return data, time
if __name__ == "__main__":
    # Generate a synthetic AR(10) process
    signal_sampled, time_sampled = load_wave('audio/ch_sound.wav', True)
    signal_original, time_original = load_wave('audio/ch_sound.wav', False)
    #true_ar_coeffs = np.array([0.8, -0.6, 0.4, -0.2, 0.1,-0.05, 0.025, -0.0125, 0.00625, -0.003125])
    
    #signal = np.zeros(n)

    #for i in range(10, n):
        #signal[i] = np.dot(true_ar_coeffs, signal[i-order:i][::-1]) + noise[i]
    
    # Plot generated signal
    # n = np.linspace(0,160,160)
    # plt.figure(figsize=(10, 6))
    # plt.plot(n, signal)#
    # plt.show()
    # Test yule_walker function
    order = 10
    estimated_ar_coeffs, noise_variance = yule_walker(signal_sampled, order)
    ar_coeffs_original, noise_variance_original = yule_walker(signal_original, order)

    # Test bootstrap_ar function
    B = 1000
    bootstrap_coeffs, sigma, spectrum = bootstrap_ar(signal_sampled, B, order)
    print("Bootstrap AR Coefficients (first 5):\n", bootstrap_coeffs[:5])
    lower_bound_ar, upper_bound_ar, median_ar = calculate_confidence_intervals(bootstrap_coeffs)
    
    mean_ar = np.mean(bootstrap_coeffs, axis=0)
    lower_bound_spectrum, upper_bound_spectrum, median_spectrum = calculate_confidence_intervals(spectrum)
    mean_spectrum = np.mean(spectrum, axis=0)
    # mean_sigma = np.mean(sigma, axis=0)
    # lower_bound_sigma, upper_bound_sigma, median_sigma = calculate_confidence_intervals(sigma)

    # mean_spectrum = calculate_spectrum(mean_ar, mean_sigma)
    # upper_spectrum = calculate_spectrum(upper_bound_ar, upper_bound_sigma)
    # lower_spectrum = calculate_spectrum(lower_bound_ar, lower_bound_sigma)
    
    # 
    # mean_spectrum_2 = np.mean(spectrum, axis=0)
    # ar_coeffs_orig, variance_orig = yule_walker(signal, order)
    # spectrum_orig = calculate_spectrum(ar_coeffs_orig, variance_orig)
    # plt.figure(figsize=(10,6))
    # N = len(mean_spectrum)
    # n = np.linspace(0,1,N)
    # plt.plot(n, spectrum_orig, label='mean', linestyle='-')
    # plt.plot(n, upper_bound_spectrum, label='upper', linestyle='--')
    # plt.plot(n, lower_bound_spectrum, label='lower', linestyle='--')
    # plt.title('Bootstrap Spectrum, Confidence Interval of SPectrum')
    # plt.legend()
    # plt.show()


    # # Plot results
    # plt.figure(figsize=(10,6))
    # N = len(mean_spectrum)
    # n = np.linspace(0,1,N)
    # plt.plot(n, mean_spectrum, label='mean', linestyle='-')
    # plt.plot(n, upper_spectrum, label='upper', linestyle='--')
    # plt.plot(n, lower_spectrum, label='lower', linestyle='--')
    # plt.title('Bootstrap Spectrum, calculated with Confidence Interval of AR-coefficients')
    # plt.legend()
    # plt.show()


    




    plt.figure(figsize=(10, 6))

    # Go through all AR-coefficients
    for i in range(len(bootstrap_coeffs[1,:])):
        if i == 0:
            plt.plot([i, i], [lower_bound_ar[i], upper_bound_ar[i]], color='blue', lw=2, label='confidence interval')  # Konfidenzintervall
            plt.scatter(i, median_ar[i], color='green', marker='x', label='median')  # Durchschnitt der Bootstrap-Koeffizienten
            plt.scatter(i,ar_coeffs_original[i], color='blue', marker='x', label='original') # AR Coeffs estimated from original data
        else:
            plt.plot([i, i], [lower_bound_ar[i], upper_bound_ar[i]], color='blue', lw=2)  # Konfidenzintervall
            plt.scatter(i, median_ar[i], color='green', marker='x')  # Durchschnitt der Bootstrap-Koeffizienten
            plt.scatter(i,ar_coeffs_original[i], color='blue', marker='x') # AR Coeffs estimated from original data

    # Plot Confidence Interval
    plt.xticks(np.arange(10), [f"AR{i+1}" for i in range(10)])
    #plt.ylim([-1.5,1])
    plt.xlabel("AR-Coefficients")
    plt.ylabel("Value")
    plt.title("Confidence Interval of AR-Coefficients")
    plt.legend()
    plt.grid(True)
    
    plt.figure(figsize=(10, 6))

    # Go through all AR-coefficients
    for i in range(len(bootstrap_coeffs[1,:])):
        if i == 0:
            plt.plot([i, i], [lower_bound_ar[i], upper_bound_ar[i]], color='blue', lw=2, label='confidence interval')  # Konfidenzintervall
            plt.scatter(i, mean_ar[i], color='green', marker='x', label='mean')  # Durchschnitt der Bootstrap-Koeffizienten
            plt.scatter(i,ar_coeffs_original[i], color='yellow', marker='s', label='original') # AR Coeffs estimated from original data
        else:
            plt.plot([i, i], [lower_bound_ar[i], upper_bound_ar[i]], color='blue', lw=2)  # Konfidenzintervall
            plt.scatter(i, mean_ar[i], color='green', marker='x')  # Durchschnitt der Bootstrap-Koeffizienten
            plt.scatter(i,ar_coeffs_original[i], color='yellow', marker='s') # AR Coeffs estimated from original data
    # Plot Confidence Interval
    plt.xticks(np.arange(10), [f"AR{i+1}" for i in range(10)])
    #plt.ylim([-1.5,1])
    plt.xlabel("AR-Coefficients")
    plt.ylabel("Value")
    plt.title("Confidence Interval of AR-Coefficients")
    plt.legend()
    plt.grid(True)
    

    plt.figure(figsize=(10, 6))
    n = np.linspace(0,1,81)
    plt.plot(n, 10*np.log10(np.abs(upper_bound_spectrum)),label="upper bound", linestyle="--", color="black")
    plt.plot(n, 10*np.log10(np.abs(lower_bound_spectrum)), label="lower bound", linestyle="-.", color="black")
    plt.plot(n, 10*np.log10(np.abs(median_spectrum)), label="median", linestyle= "-", color="blue")
    plt.grid(True)
    plt.legend()
    plt.ylabel("C_xx [dB]")
    plt.xlabel("w/pi")
    plt.title('Confidence Bounds of Spectrum')

    plt.figure(figsize=(10, 6))
    n = np.linspace(0,1,81)
    plt.plot(n, 10*np.log10(np.abs(upper_bound_spectrum)),label="upper bound", linestyle="--", color="black")
    plt.plot(n, 10*np.log10(np.abs(lower_bound_spectrum)), label="lower bound", linestyle="-.", color="black")
    plt.plot(n, 10*np.log10(np.abs(mean_spectrum)), label="mean", linestyle= "-", color="blue")
    plt.grid(True)
    plt.legend()
    plt.ylabel("C_xx [dB]")
    plt.xlabel("w/pi")
    plt.title('Confidence Bounds of Spectrum')


    plt.figure(figsize=(10, 6))
    for i in range(B):
        plt.plot(n,10*np.log10(np.abs(spectrum[i,:])))
        

    plt.show()
    x = 5

