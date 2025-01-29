#!/usr/bin/env python
# coding: utf-8

import numpy as np
from scipy.signal import resample
from scipy.signal import wiener
import matplotlib.pyplot as plt
import paper_teil_1

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
    # Bootrap x and x~ at the same time
    ar_coeffs_original, variance_orig, spectrum_orig = paper_teil_1.bootstrap_ar(original_signal, 1000, 10)
    ar_coeffs_processed, variance_processed, spectrum_processed = paper_teil_1.bootstrap_ar(processed_signal, 1000, 10)
    lower_bound_spec_orig, upper_bound_spec_orig, median_spec_orig = paper_teil_1.calculate_confidence_intervals(spectrum_orig)
    lower_bound_spec_proc, upper_bound_spec_proc, median_spec_proc = paper_teil_1.calculate_confidence_intervals(spectrum_processed)
    V_star = np.empty((1000, 80))
    rms_star = np.empty(1000)
    for i in range(1000):
        for k in range(80):
            # Calculate log spectral distance for each bootsrap sample
            V_star[i,k] = 10 * (np.log10(spectrum_orig[i,k])- np.log10(spectrum_processed[i,k]))
        rms_star[i] = np.sqrt(np.sum(pow(np.abs(V_star[i,:]),2)))
    lower_bound_V, upper_bound_V, median_V = paper_teil_1.calculate_confidence_intervals(V_star)
    mean_V = np.mean(V_star, axis=0)
    n = np.linspace(0,1,80)
    n2 = np.linspace(0,1,81)
    plt.figure()
    plt.plot(n,lower_bound_V,linestyle='--', label='lower bound')
    plt.plot(n,mean_V, linestyle='-', label='median')
    plt.plot(n,upper_bound_V, linestyle='dashdot', label='upper bound')
    plt.ylabel('V(e^jw) [dB]')
    plt.xlabel('w/pi')
    #plt.ylim([-10,15])
    plt.title('Spectral Difference Confidence Interval, calculated all V')
    plt.legend()
    plt.figure()
    plt.plot(n2,10*(np.log10(lower_bound_spec_orig)-np.log10(lower_bound_spec_proc)),linestyle='--', label='lower bound')
    plt.plot(n2,10*(np.log10(median_spec_orig)-np.log10(median_spec_proc)), linestyle='-', label='median')
    plt.plot(n2,10*(np.log10(upper_bound_spec_orig)-np.log10(upper_bound_spec_proc)), linestyle='dashdot', label='upper bound')
    #plt.ylim([-10,15])
    plt.ylabel('V(e^jw) [dB]')
    plt.xlabel('w/pi')
    plt.title('Spectral Difference Confidence Interval, calculated conf Int spectrum')
    plt.legend()
    plt.figure()
    weight = np.full(1000, 1/1000)
    rms_median = np.median(rms_star)
    plt.hist(rms_star, bins=17, color='white', edgecolor='blue', weights=weight, alpha=0.7)
    plt.scatter(rms_median,0.1,marker="x",color="red")
    plt.title('RMS of log spectral distance')
    plt.ylabel('Rel. frequency')
    plt.xlabel('RMS of log spectral distance, d_2 [dB]')
    plt.show()
    #TODO: Clean code

    
    
if __name__ == "__main__":
    ch_sound,_ = paper_teil_1.load_wave('audio/ch_sound.wav',True, plot=True, normalize=True)
    noise,_ = paper_teil_1.load_wave('audio/vehicle-movement-noise.wav',True, plot=True, normalize=True)
    

    snr = 3
    noisy_signal = contaminate_signal_with_noise(ch_sound, noise, snr)
    
    plt.figure(figsize=(10, 6))
    plt.plot(np.linspace(0,160,160), ch_sound)
    plt.title('Speech')
    
    plt.figure(figsize=(10, 6))
    plt.plot(np.linspace(0,160,160), noise)
    plt.title('noise')

    plt.figure(figsize=(10, 6))
    plt.plot(np.linspace(0,160,160), noisy_signal)
    plt.title('noisy signal')
    
    processed_signal = wiener(noisy_signal)
    plt.figure(figsize=(10, 6))
    plt.plot(np.linspace(0,160,160), processed_signal)
    plt.title('processed_signal')
    
    
    spectral_difference_bootstrap(ch_sound,processed_signal)
    plt.show()