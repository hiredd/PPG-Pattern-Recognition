import sys
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter, freqz, filtfilt, welch, medfilt
from scipy.fftpack import fft
from datetime import datetime
sys.path.append('lib')
import detect_peaks
from sklearn import preprocessing
import matplotlib.pyplot as plt
from peakutils.peak import indexes

class Signal:

    def __init__(self, signal, timestamps, fs = 80):
        self.timestamps = np.array(timestamps)
        self.sample_freq = fs
        self.content = np.array(signal)
        self.correct_saturation()
        self.remove_outliers()
        self.content = self.highpass_filter(1)

    def extract_features(self, validate_HR_range=True):
        feature_vector = []

        f, psd = self.log_PSD()

        i_freq = self.i_freq

        # F0: Mean amplitude of HF components (>20Hz)
        feature_vector.append(abs(np.mean(psd[i_freq(f,20):])))

        # F1: HF/LF component magnitude ratio (4-5Hz)/1-2Hz
        feature_vector.append(abs(
            np.sum(psd[i_freq(f,4):i_freq(f,10)+1]) /\
            np.sum(psd[i_freq(f,1):i_freq(f,2)+1])))

        # F2: VLF/LF component magnitude ratio (<1Hz)/1-2Hz
        feature_vector.append(abs(
            np.sum(psd[i_freq(f,0):i_freq(f,1)+1]) /\
            np.sum(psd[i_freq(f,1):i_freq(f,2)+1])))

        # F3: Peak variance
        BPM, peak_variance = self.get_BPM_and_peak_variance()
        feature_vector.append(peak_variance)

        if validate_HR_range and (BPM < 50 or BPM > 120):
            return None

        return feature_vector

    def extract_PSD_features(self):
        freqs, psd = self.log_PSD()
        # For classification, we don't care about the PPG frequency, just that 
        # it's in a normal human range, so take the mean of that frequency range.
        startf = self.i_freq(freqs, 1)
        endf = self.i_freq(freqs, 2.5)+1
        a = np.array([psd[0]]) 
        b = np.array([np.mean(psd[startf:endf])]) 
        c = psd[4:]
        psd = np.concatenate([a,b,c])
        return psd

    # Correct signal saturation by scaling readings by the maximum amplitude.
    def correct_saturation(self):
        max_val = np.max(self.content)
        signal_diff = np.diff(self.content)
        signal_diff_end = len(signal_diff)-1
        max_slope = 50000 # saturation (positive or negative)
        
        for k in range(len(signal_diff)):
            if signal_diff[k] > max_slope:
                # Pull subsequent values down
                self.content[k+1:signal_diff_end] = self.content[k+1:signal_diff_end] - max_val
            if signal_diff[k] < -1*max_slope:
                # Pull subsequent values up
                self.content[k+1:signal_diff_end] = self.content[k+1:signal_diff_end] + max_val

    def highpass_filter(self, cutoff, order=2):
        nyq = 0.5 * self.sample_freq
        cutoff = cutoff / nyq
        b, a = butter(order, cutoff, btype='highpass', analog=False)
        return filtfilt(b, a, self.content.tolist())

    def bandpass_filter(self, cutoff_low, cutoff_high, order=2):
        nyq = 0.5 * self.sample_freq
        normal_cutoff_low = cutoff_low / nyq
        normal_cutoff_high = cutoff_high / nyq
        b, a = butter(order, [normal_cutoff_low, normal_cutoff_high], btype='bandpass', analog=False)
        return filtfilt(b, a, self.content.tolist())

    # Moving median filter
    def remove_outliers(self, window = 3):
        self.content = np.array(medfilt(self.content.tolist(), window))

    # Log power spectral density of the signal segment
    # *Input: sample_window (int): length of overlapping segments for pwelch averaging
    # *Output: Tuple of frequency spectrum list and corresponding scaled log base 10
    # of the PSD of the signal segment start-end
    def log_PSD(self, sample_window = 128):
        segment = self.content
        f, psd = welch(segment, fs = self.sample_freq, nperseg = sample_window)
        return f, np.log10(psd)

    # Calculate the heart rate and the beat variance 
    def get_BPM_and_peak_variance(self):
        # Focus on standard HR frequency range
        segment = self.bandpass_filter(0.8, 2.5)
        indices = detect_peaks.detect_peaks(segment, mph = 50)

        peak_variance = np.finfo(np.float32).max
        if len(indices) > 1:
            peak_variance = np.var(np.diff(indices))

        time_difference = self.timestamp_in_datetime(-1) - self.timestamp_in_datetime(0)
        time_difference_in_minutes = float(time_difference.seconds + 
                                     float(time_difference.microseconds)/10**6)/60.0
        BPM = float(len(indices)) / time_difference_in_minutes

        return BPM, peak_variance

    def timestamp_in_datetime(self, timestamp_id):
        try:
            timestamp = datetime.strptime(self.timestamps[timestamp_id], '%Y-%m-%d %H:%M:%S.%f')
        except ValueError:
            timestamp = datetime.strptime(self.timestamps[timestamp_id], '%Y-%m-%d %H:%M:%S')
        return timestamp

    def length(self):
        return np.shape(self.content)[0]

    # Return index of the closest frequency in the given frequency range
    def i_freq(self, frequency_range_list, frequency):
        i = 0
        while i < len(frequency_range_list) and frequency_range_list[i] < frequency:
            i += 1
        if i == 0:
            return 0
        return i-1