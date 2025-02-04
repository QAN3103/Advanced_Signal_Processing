#!/usr/bin/env python
# coding: utf-8

import numpy as np
from scipy.signal import resample
from scipy.signal import wiener
import matplotlib.pyplot as plt
import paper_teil_1 as pp1

def wiener_filter(noisy_signal):
    
    return wiener(noisy_signal)

def contaminate_signal_with_noise(signal, noise, snr):
    twos = np.full(len(signal), 2)
    rms_speech = np.sqrt(np.mean(pow(signal,twos)))
    rms_noise = np.sqrt(np.mean(pow(noise,twos)))
    # Gewünschtes SNR in dB
    target_snr_db = snr

    # Faktor zur Skalierung des Störsignals berechnen
    snr_linear = 10**(target_snr_db / 20)
    scaling_factor = rms_speech / (snr_linear * rms_noise)

    # Skaliere das Störsignal
    noise_scaled = noise * scaling_factor
    return signal + noise_scaled

def spectral_difference_bootstrap(original_signal, processed_signal):
    l = np.arange(0,10)
    # Bootrap x and x~ at the same time
    ar_coeffs_original, parcor_coeffs_original, variance_orig, spectrum_orig = pp1.bootstrap_ar(original_signal, 1000, 10)
    ar_coeffs_processed, parcor_coeffs_processed, variance_processed, spectrum_processed = pp1.bootstrap_ar(processed_signal, 1000, 10)
    # Confidence Bounds of Spectrum, AR and Parcor
    lower_bound_spec_orig, upper_bound_spec_orig, median_spec_orig = pp1.calculate_confidence_intervals(spectrum_orig)
    lower_bound_spec_proc, upper_bound_spec_proc, median_spec_proc = pp1.calculate_confidence_intervals(spectrum_processed)

    lower_bound_ar_orig, upper_bound_ar_orig, median_ar_orig = pp1.calculate_confidence_intervals(ar_coeffs_original)
    lower_bound_ar_proc, upper_bound_ar_proc, median_ar_proc = pp1.calculate_confidence_intervals(ar_coeffs_processed)
    
    lower_bound_parcor_orig, upper_bound_parcor_orig, median_parcor_orig = pp1.calculate_confidence_intervals(parcor_coeffs_original)
    lower_bound_parcor_proc, upper_bound_parcor_proc, median_parcor_proc = pp1.calculate_confidence_intervals(parcor_coeffs_processed)
    
    V_star = np.empty((1000, 80))
    rms_star = np.empty(1000)
    for i in range(1000):
        for k in range(80):
            # Calculate log spectral distance for each bootsrap sample
            V_star[i,k] = 10 * (np.log10(spectrum_orig[i,k])- np.log10(spectrum_processed[i,k]))
        rms_star[i] = np.sqrt((1/80)*np.sum(pow(np.abs(V_star[i,:]),2)))
    lower_bound_V, upper_bound_V, median_V = pp1.calculate_confidence_intervals(V_star)
    ### Save data as textfiles
    k = np.arange(1,1001)
    txt = np.column_stack((k,rms_star))
    np.savetxt("txtfiles/rms.dat",txt,fmt="%f", delimiter="\t",header="index\trms", comments="")
    num_bins = 17
    hist, bin_edges = np.histogram(rms_star,bins=num_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:])/2
    hist_matrix = np.column_stack((hist/1000,bin_centers))
    print(hist_matrix)
    np.savetxt("txtfiles/rms.dat",hist_matrix,fmt="%f", delimiter="\t",header="freq\trms", comments="")
    rms_median = np.median(rms_star)
    med_rms = np.column_stack((0.1,rms_median))
    np.savetxt("txtfiles/rms_median.dat",med_rms,fmt="%f", delimiter="\t",header="freq\trms", comments="")

    ###
    mean_V = np.mean(V_star, axis=0)
    n = np.linspace(0,1,80)
    n2 = np.linspace(0,1,81)
    ### Figure 1: Spectral Difference
    plt.figure(figsize=(10, 6))
    plt.plot(n,lower_bound_V,linestyle='--', label='lower bound', color='black')
    plt.plot(n,median_V, linestyle='-', label='median', color='blue')
    plt.plot(n,upper_bound_V, linestyle='dashdot', label='upper bound', color='black')
    plt.ylabel(r"$V(e^{j\omega})$ [dB]", fontsize=12)

    plt.xlabel(r"$\omega/\pi$", fontsize=12)
    #plt.ylim([-10,15])
    plt.title('Spectral Difference Confidence Interval, calculated all V')
    plt.legend(fontsize=12)
    
    ### Figure 2: RMS
    plt.figure(figsize=(10, 6))
    weight = np.full(1000, 1/1000)
    
    plt.hist(rms_star, bins=17, color='white', edgecolor='blue', weights=weight, alpha=0.7)
    plt.scatter(rms_median,0.1,marker="x",color="red")
    plt.title('RMS of log spectral distance')
    plt.ylabel('Rel. frequency', fontsize=12)
    plt.xlabel(r"RMS of log spectral distance, $d_2$ [dB]", fontsize=12)

    ### Figure 3: Ar Coeffs orig vs processed
    plt.figure(figsize=(10, 6))
    low = lower_bound_ar_orig
    high = upper_bound_ar_orig
    med = median_ar_orig
    y_err_low = np.empty(len(low))
    y_err_high = np.empty(len(high))
    for i in range(len(low)):
        y_err_low[i] = med[i] - low[i]
        y_err_high[i] = high[i] - med[i] 
    plt.errorbar(l,med,[y_err_low, y_err_high], fmt='x', color='black', ecolor='blue', capsize=4, label=r'$a_k$', alpha=0.7)
    low = lower_bound_ar_proc
    high = upper_bound_ar_proc
    med = median_ar_proc
    y_err_low = np.empty(len(low))
    y_err_high = np.empty(len(high))
    for i in range(len(low)):
        y_err_low[i] = med[i] - low[i]
        y_err_high[i] = high[i] - med[i] 
    plt.errorbar(l,med,[y_err_low, y_err_high], fmt='x', color='red', ecolor='magenta', capsize=4, label=r'$\tilde{a}_k$', alpha=0.7)
    plt.title("Comparison of AR-Coefficients")
    plt.legend(fontsize=12)
    plt.xlabel(r'$k$', fontsize=12)
    plt.ylabel(r'$a_k$', fontsize=12)
    plt.xticks(np.arange(10), [i+1 for i in range(10)])
    plt.grid(True)
    plt.show()

    
    
if __name__ == "__main__":
    ch_sound,_ = pp1.load_wave('audio/ch_sound.wav',True, plot=False, normalize=True)
    noise,_ = pp1.load_wave('audio/vehicle-movement-noise.wav',True, plot=False, normalize=True)
    

    snr = 3
    noisy_signal = contaminate_signal_with_noise(ch_sound, noise, snr)
    
    plt.figure(figsize=(10, 6))
    plt.plot(np.linspace(0,160,160), ch_sound)
    plt.title('Speech')
    plt.xlabel('n', fontsize=12)
    plt.ylabel('x(n)', fontsize=12)
    
    plt.figure(figsize=(10, 6))
    plt.plot(np.linspace(0,160,160), noise)
    plt.title('Noise')
    plt.xlabel('n', fontsize=12)
    plt.ylabel('v(n)', fontsize=12)

    plt.figure(figsize=(10, 6))
    plt.plot(np.linspace(0,160,160), noisy_signal)
    plt.title('Noisy Signal, SNR=3dB')
    plt.xlabel('n', fontsize=12)
    plt.ylabel('v(n) + x(n)', fontsize=12)
    
    processed_signal = wiener(noisy_signal)
    plt.figure(figsize=(10, 6))
    plt.plot(np.linspace(0,160,160), processed_signal)
    plt.title('Processed Signal')
    plt.xlabel('n', fontsize=12)
    plt.ylabel(r'$\tilde{x}(n)$', fontsize=12)
    
    
    spectral_difference_bootstrap(ch_sound,processed_signal)
    plt.show()