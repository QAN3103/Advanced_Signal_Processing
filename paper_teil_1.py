#!/usr/bin/env python
# coding: utf-8

from math import pi
from msilib.schema import Media
from scipy.linalg import toeplitz
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import resample
from scipy.signal import lfilter
import paper_teil_2 as p2
from statsmodels.regression.linear_model import yule_walker as yw
from statsmodels.regression.linear_model import burg
import parcor



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
    ar_coeffs, noise_variance = yw(signal,order)
    poly = np.array([1]) + list(ar_coeffs)
    poles = np.abs(np.roots(poly))
    #if (np.any(poles) > 1): print("ar coeffs unstable")
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
    Calculates the new signal, based on original signal and residuals.
    
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
        signal_star[n] = -(-estimate + residuals[n - order]) 
    
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
    parcor_coeffs_star_all = np.zeros((B, len(ar_coeffs_orig)))
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
        # Step 2.1 centering data
        signal_star = signal_star - np.mean(signal_star)
        
            
        # Step 3        
        ar_coeffs_star, variance_star = yule_walker(signal_star, order)
        parcor_coeffs_star = parcor.compute_parcor(signal_star,order)
        
        # Save all Bootstrap estimates
        ar_coeffs_star_all[i] = ar_coeffs_star
        variance_all[i] = variance_star
        parcor_coeffs_star_all[i] = parcor_coeffs_star
        
        # Calculate Spectrum of each Bootstrap estimate
        spectrum = calculate_spectrum(-ar_coeffs_star, variance_star)
        spectrum_all[i] = spectrum
        
    return ar_coeffs_star_all, parcor_coeffs_star_all, variance_all, spectrum_all

def calculate_confidence_intervals(bootstrap_coeffs, alpha=0.05):
    """ 
    Calculates Confidence Bounds and Median
    """
    lower_bound = np.percentile(bootstrap_coeffs, 100 * (alpha / 2), axis=0)
    upper_bound = np.percentile(bootstrap_coeffs, 100 * (1 - alpha / 2), axis=0)
    median = np.percentile(bootstrap_coeffs, 50, axis=0)
    return lower_bound, upper_bound, median

def calculate_spectrum(ar_coeffs, variance, N=160):
    """
    Calculates the spectrum as described in paper
    """
    k = np.arange(0, N // 2 + 1)  # Create an array of k from 0 to N/2
    omega = 2 * np.pi * k / N   # Calculate ωk for each k
    Cxx = np.empty(len(omega))
    for i in range(len(omega)):
        sum = 0
        for k in range(len(ar_coeffs)):
            sum = sum + ar_coeffs[k] * np.exp(-omega[i] * (k + 1) * 1j)
        Cxx[i] = (1/(2*pi))*(variance / pow(np.abs(1+ sum), 2))
    return Cxx

def load_wave(filepath, downsample, plot, normalize):
    """
    Loads wave file and returns discrete signal
    """
    samplerate, data = wavfile.read(filepath)
    time = np.linspace(0, len(data) / samplerate, num=len(data))
    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(time, data)
        plt.title('Original Timeseries')
        plt.xlabel('Zeit [s]')
        plt.ylabel('Amplitude')
        plt.grid()
    if downsample:
        data = resample(data,160)
        time = resample(time,160)
        if normalize:
            max_value = np.max(np.abs(data))
            data = data /(100*max_value)
        if plot:
            plt.figure(figsize=(10, 5))
            plt.plot(np.linspace(1,160,160), data)
            plt.title('Speech Signal')
            plt.xlabel('n')
            plt.ylabel('x(n)')
            plt.grid()
            ind = np.arange(1,len(data)+1)
            txt = np.column_stack((ind,data))
            np.savetxt("txtfiles/speech_signal.dat", txt, fmt="%f", delimiter="\t",header="Index\tWert", comments="")
    else:
        N = len(data)
        data = resample(data,N)
        time = resample(time, N)
        if normalize:
            max_value = np.max(np.abs(data))
            data = data /(100*max_value)
        if plot:
            plt.figure(figsize=(10, 5))
            plt.plot(np.linspace(1, N,N), data)
            plt.title('Resampled Timeseries (N=high default)')
            plt.xlabel('n')
            plt.ylabel('x(n)')
            plt.grid()
    #if plot: plt.show()

    return data, time

def generate_ar_process(ar_coeffs, N, noise_var=1):
    """
    Generates discrete time signal of an AR(p) process
    
    Parameter:
    - ar_coeffs: Array der AR-Koeffizienten (p Elemente)
    - N: Anzahl der zu simulierenden Punkte
    - noise_var: Varianz des weißen Rauschens e(n)
    
    Rückgabe:
    - x: Generierte AR-Zeitreihe
    """
    p = len(ar_coeffs)  # Ordnung des AR-Prozesses
    x = np.zeros(N)     # Initialisiere Zeitreihe
    noise = np.random.normal(0, np.sqrt(noise_var), N)  # Weißes Rauschen e(n)

    # Initialisierung mit Zufallswerten für die ersten p Werte
    x[:p] = np.random.normal(0, 1, p)

    # Rekursive Berechnung von x(n) nach der AR-Gleichung
    for n in range(p, N):
        x[n] = -np.dot(ar_coeffs, x[n-p:n][::-1]) + noise[n]  # x(n) = -sum(a_k * x(n-k)) + e(n)

    return x

def calculate_coverage(true_ar_coeffs, ar_signal,B):
    order = len(true_ar_coeffs)
    is_covered = np.empty([B,order])
    for i in range(B):
        ar_coeffs_boot,_,_ = bootstrap_ar(ar_signal,1000,order)
        lower,upper,_=calculate_confidence_intervals(ar_coeffs_boot)
        for k in range(order):
            if(lower[k] <= true_ar_coeffs[k] <= upper[k]):
                is_covered[i,k] = 1
            else:
                is_covered[i,k] = 0
    coverage = np.empty(order)
    for i in range(order):
        coverage[i] = np.sum(is_covered[:,i])/B
    return coverage        



    
if __name__ == "__main__":
    # Load wave files and set general variables
    order = 10
    B = 1000
    k = np.arange(1,11)
    z = 0
    while z<2:
        if z==0:
            signal_sampled, time_sampled = load_wave('audio/a_sound.wav', downsample=True, plot=True, normalize=True)
            signal_original, time_original = load_wave('audio/a_sound.wav', downsample=False, plot=False, normalize=True)
        if z==1:
            signal_sampled, time_sampled = load_wave('audio/ch_sound.wav', downsample=True, plot=True, normalize=True)
            signal_original, time_original = load_wave('audio/ch_sound.wav', downsample=False, plot=False, normalize=True)
        z=z+1
    
        # true_ar_coeffs = np.array([0.5, -0.3, 0.2, -0.1, 0.05, -0.03, 0.02, -0.01, 0.005, -0.002  ])
        # poly = np.array([1] + list(-true_ar_coeffs))
        # poles = np.abs(np.roots(poly))

    

        # Yule-Walker
        # estimated_ar_coeffs, noise_variance = yule_walker(signal_sampled, order)
        ar_coeffs_original, noise_variance_original = yule_walker(signal_original, order)

        # Bootstrap
        bootstrap_ar_coeffs, bootstrap_parcor_coeffs, sigma, spectrum = bootstrap_ar(signal_sampled, B, order)
        print("Bootstrap AR Coefficients (first 5):\n", bootstrap_ar_coeffs[:5])
        # Calculate Confidence Bounds of Spectrum and AR Coeffs
        lower_bound_ar, upper_bound_ar, median_ar = calculate_confidence_intervals(bootstrap_ar_coeffs)
        #mean_ar = np.mean(bootstrap_ar_coeffs, axis=0)
        lower_bound_parcor, upper_bound_parcor, median_parcor = calculate_confidence_intervals(bootstrap_parcor_coeffs)
        lower_bound_spectrum, upper_bound_spectrum, median_spectrum = calculate_confidence_intervals(spectrum)
        #mean_spectrum = np.mean(spectrum, axis=0)

    
        ### Figure 1 AR-Coefficients Conf Interval + Median
        plt.figure(figsize=(10, 6))
        low = lower_bound_ar
        high = upper_bound_ar
        med = median_ar
        y_err_low = np.empty(len(low))
        y_err_high = np.empty(len(high))
        for i in range(len(low)):
            y_err_low[i] = med[i] - low[i]
            y_err_high[i] = high[i] - med[i]
        txt = np.column_stack((k,med,y_err_low,y_err_high))
        np.savetxt("txtfiles/ar_conf_a.dat", txt, fmt="%f", delimiter="\t",header="Index\tMedian\terrorlow\terrorhigh", comments="")
        # Go through all AR-coefficients
        # Plot Confidence Interval
        plt.errorbar(k,med,[y_err_low, y_err_high], fmt='x', color='black', ecolor='blue', capsize=4, label='confidence int.')
        plt.scatter([], [], marker='x', color='black', label='median')
        plt.xticks(np.arange(10), [i+1 for i in range(10)])
        plt.ylim([-1.5,1])
        # plt.xlabel("AR-Coefficients")
        # plt.ylabel("Value")
        plt.title("Confidence Interval of AR-Coefficients")
        #plt.legend()
        plt.xlabel(r'$k$', fontsize=12)
        plt.ylabel(r'$a_k$', fontsize=12)
        plt.grid(True)

        ### Figure 2 Parcor-Coefficients Conf Interval + Median
        plt.figure(figsize=(10, 6))
        low = lower_bound_parcor
        high = upper_bound_parcor
        med = median_parcor
        y_err_low = np.empty(len(low))
        y_err_high = np.empty(len(high))
        for i in range(len(low)):
            y_err_low[i] = med[i] - low[i]
            y_err_high[i] = high[i] - med[i] 
        # Go through all AR-coefficients
        # Plot Confidence Interval
        plt.errorbar(k,med,[y_err_low, y_err_high], fmt='x', color='black', ecolor='blue', capsize=4, label='confidence int.')
        plt.scatter([], [], marker='x', color='black', label='median')
        plt.xticks(np.arange(10), [i+1 for i in range(10)])
        #plt.ylim([-1.5,1])
        # plt.xlabel("AR-Coefficients")
        # plt.ylabel("Value")
        plt.title("Confidence Interval of Parcor-Coefficients")
        #plt.legend()
        plt.xlabel(r'$k$', fontsize=12)
        plt.ylabel(r'$r_k$', fontsize=12)
        plt.grid(True)
    

    
        """
        ### Figure 2 AR-Coefficients Conf Interval + Median
        # Go through all AR-coefficients
        # Plot Confidence Interval
        plt.figure(figsize=(10, 6))
        for i in range(len(bootstrap_ar_coeffs[1,:])):
            if i == 0:
                plt.plot([i, i], [lower_bound_ar[i], upper_bound_ar[i]], color='blue', lw=2, label='confidence interval')  # Konfidenzintervall
                plt.scatter(i, mean_ar[i], color='green', marker='x', label='mean')  # Durchschnitt der Bootstrap-Koeffizienten
                plt.scatter(i,ar_coeffs_original[i], color='yellow', marker='s', label='original') # AR Coeffs estimated from original data
            else:
                plt.plot([i, i], [lower_bound_ar[i], upper_bound_ar[i]], color='blue', lw=2)  # Konfidenzintervall
                plt.scatter(i, mean_ar[i], color='green', marker='x')  # Durchschnitt der Bootstrap-Koeffizienten
                plt.scatter(i,ar_coeffs_original[i], color='yellow', marker='s') # AR Coeffs estimated from original data
        plt.xticks(np.arange(10), [f"AR{i+1}" for i in range(10)])
        plt.ylim([-1.5,1])
        plt.xlabel("AR-Coefficients")
        plt.ylabel("Value")
        plt.title("Confidence Interval of AR-Coefficients")
        plt.legend()
        plt.grid(True)
        """
        ### Figure 3: Confidence Bounds of Spectrum + Median
        plt.figure(figsize=(10, 6))
        n = np.linspace(0,1,81)
        plt.plot(n, 10*np.log10(np.abs(upper_bound_spectrum)),label="upper bound", linestyle="--", color="black")
        plt.plot(n, 10*np.log10(np.abs(lower_bound_spectrum)), label="lower bound", linestyle="-.", color="black")
        plt.plot(n, 10*np.log10(np.abs(median_spectrum)), label="median", linestyle= "-", color="blue")
        plt.grid(True)
        plt.legend(fontsize=12)
        plt.ylabel(r"$C_{xx}(e^{j\omega})$", fontsize=12)
        plt.xlabel(r"$\omega/\pi$", fontsize=12)
        plt.title('Confidence Bounds of Spectrum')

        txt = np.column_stack((n,10*np.log10(abs(median_spectrum)),10*np.log10(abs(lower_bound_spectrum)),10*np.log10(abs(upper_bound_spectrum))))
        np.savetxt("txtfiles/spectrum_conf_a.dat", txt, fmt="%f", delimiter="\t",header="Index\tmedian\tlowerbound\tupperbound", comments="")

        """
        ### Figure 4: Confidence Bounds of Spectrum + Mean
        plt.figure(figsize=(10, 6))
        n = np.linspace(0,1,81)
        plt.plot(n, 10*np.log10(np.abs(upper_bound_spectrum)),label="upper bound", linestyle="--", color="black")
        plt.plot(n, 10*np.log10(np.abs(lower_bound_spectrum)), label="lower bound", linestyle="-.", color="black")
        plt.plot(n, 10*np.log10(np.abs(mean_spectrum)), label="mean", linestyle= "-", color="blue")
        plt.grid(True)
        plt.legend()
        plt.ylabel("C_xx [dB]")
        plt.xlabel("w/pi")
        plt.title('Confidence Bounds of Spectrum; a-sound')
        """

        ### test Bootstrap on synthetic Signal
        ar_signal = generate_ar_process(-median_ar, 160)
        #coverage = calculate_coverage(median_ar,ar_signal,100)
        ar_coeffs_firtsestimate, variance_syn = yule_walker(ar_signal,10)
        spectrum_syn = calculate_spectrum(median_ar, variance_syn)
        plt.figure(figsize=(10, 6))
        n = np.linspace(0,160,160)
        ### Figure 5: synthetic Signal
        plt.plot(n, ar_signal)
        plt.title("Synthetic AR-process")
        plt.ylabel('x(n)', fontsize=12)
        plt.xlabel('n', fontsize=12)
        ### Figure 6: Spectrum of synthetic signal
        plt.figure()
        w = np.linspace(0,1,81)
        plt.plot(w,10*np.log10(spectrum_syn))
        plt.title("Spectrum of synthetic AR-process")
        plt.ylabel(r"$C_{xx}(e^{j\omega})$", fontsize=12)

        plt.xlabel(r"$\omega/\pi$", fontsize=12)

        parcor_coeffs_orig = parcor.ar_to_parcor(median_ar)

        estimated_ar_syn, estimated_parcor_syn, _, spectrum_syn = bootstrap_ar(ar_signal,1000,10)
        ar_sy_lower_bound, ar_sy_upper_bound, ar_sy_median = calculate_confidence_intervals(estimated_ar_syn)
        parcor_sy_lower_bound, parcor_sy_upper_bound, parcor_sy_median = calculate_confidence_intervals(estimated_parcor_syn)
    
        ### Figure 7: AR Coeefs Conf Interval + Median + Original of synthetic signal    
        plt.figure(figsize=(10, 6))
        low = ar_sy_lower_bound
        high = ar_sy_upper_bound
        med = ar_sy_median
        y_err_low = np.empty(len(low))
        y_err_high = np.empty(len(high))
        for i in range(len(low)):
            y_err_low[i] = med[i] - low[i]
            y_err_high[i] = high[i] - med[i] 
        # Go through all AR-coefficients
        # Plot Confidence Interval
        plt.errorbar(k,med,[y_err_low, y_err_high], fmt='x', color='black', ecolor='blue', capsize=4)
        plt.scatter([], [], marker='x', color='black', label='median')
        plt.scatter(k,median_ar, marker='x', color='red', label='original')


        plt.xticks(np.arange(10), [i+1 for i in range(10)])
        #plt.ylim([-1.5,1])
        plt.title("Confidence Interval of AR-Coefficients, Based on synthetic signal")
        plt.legend(fontsize=12)
        plt.xlabel(r'$k$', fontsize=12)
        plt.ylabel(r'$a_k$', fontsize=12)
        plt.grid(True)

        ### Figure 8: Parcor Coeefs Conf Interval + Median + Original of synthetic signal    
        plt.figure(figsize=(10, 6))
        low = parcor_sy_lower_bound
        high = parcor_sy_upper_bound
        med = parcor_sy_median
        y_err_low = np.empty(len(low))
        y_err_high = np.empty(len(high))
        for i in range(len(low)):
            y_err_low[i] = med[i] - low[i]
            y_err_high[i] = high[i] - med[i] 
        # Go through all AR-coefficients
        # Plot Confidence Interval
        plt.errorbar(k,med,[y_err_low, y_err_high], fmt='x', color='black', ecolor='blue', capsize=4)
        plt.scatter([], [], marker='x', color='black', label='median')
        plt.scatter(k,parcor_coeffs_orig, marker='x', color='red', label='original')


        plt.xticks(np.arange(10), [i+1 for i in range(10)])
        #plt.ylim([-1.5,1])
        plt.title("Confidence Interval of Parcor-Coefficients, Based on synthetic signal")
        plt.legend(fontsize=12)
        plt.xlabel(r'$k$', fontsize=12)
        plt.ylabel(r'$r_k$', fontsize=12)
        plt.grid(True)
    

    plt.show()


