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

# Separate this file into Signal (parent class), HRSignal, and AccSignal
# and put ad hoc methods in each of those classes

class Signal:

    def __init__(self, signal, timestamps, fs = 80):
        self.content = np.array(signal)
        self.timestamps = np.array(timestamps)
        self.sample_freq = fs

    def length(self):
        return np.shape(self.content)[0]

    def i_freq(self, lf, frequency):
        i = 0
        while lf[i] < frequency:
            i += 1
        if i == 0:
            return 0
        return i-1

    def extract_features(self, start, end, validate_HR_range = True):
        if start == None:
            start = 0
        if end == None:
            end = len(self.content)-1

        feature_vector = []

        f, psd = self.log_PSD(start,end)

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
        BPM, peak_variance = self.get_BPM_and_peak_variance(start, end)
        feature_vector.append(peak_variance)

        if validate_HR_range and (BPM < 50 or BPM > 120):
            return None

        return feature_vector


    # Correct signal saturation (defined as sharp slope in curve) 
    # by scaling subsequent readings by the maximum range
    # Params, Return: list of PPG readings (floats)
    def correct_saturation(self):
        max_val = np.max(self.content)
        signal_diff = np.diff(self.content)
        signal_diff_end = len(signal_diff)-1
        max_slope = 50000
        
        for k in range(len(signal_diff)):
            #print(k)
            if signal_diff[k] > max_slope:
                # Pull subsequent values down
                self.content[k+1:signal_diff_end] = self.content[k+1:signal_diff_end] - max_val
            if signal_diff[k] < -1*max_slope:
                # Pull subsequent values up
                self.content[k+1:signal_diff_end] = self.content[k+1:signal_diff_end] + max_val

    def bandpass_filter(self, cutoff_low, cutoff_high, start=None, end=None, order=2):
        nyq = 0.5 * self.sample_freq
        normal_cutoff_low = cutoff_low / nyq
        normal_cutoff_high = cutoff_high / nyq
        b, a = butter(order, [normal_cutoff_low, normal_cutoff_high], btype='bandpass', analog=False)

        if start==None and end==None:
            self.content = filtfilt(b, a, self.content.tolist())
        else:
            return filtfilt(b, a, self.content[start:end].tolist())


    # Filter outliers using a moving median filter
    def remove_outliers(self, window = 3):
        self.content = np.array(medfilt(self.content.tolist(), window))


    # Log power spectral density of the signal segment (used as a feature)
    # Input:
    #   * start (int), end (int): indeces of signal segment range
    #   * sample_window (int): length of overlapping segments for pwelch averaging
    # Output:
    #   Tuple of frequency spectrum list and corresponding scaled log base 10
    #   of the PSD of the signal segment start-end
    def log_PSD(self, start = None, end = None, sample_window = 128):
        if start == None:
            start = 0
        if end == None:
            end = len(self.content)
        segment = self.content[start:end].tolist()
        f, psd = welch(segment, fs = self.sample_freq, nperseg = sample_window)
        return f, 10*np.log10(psd)

    def fft(self, length = None):
        if length == None:
            return fft(self.content)
        return fft(self.content, length)


    # List of log PSDs of the sample_window-sized segments in the signal
    # between start and end.
    # Input:
    #   * start (int), end (int): indeces of signal segment range
    #   * sample_window (int): max-length of each segment, default to 512 ~6 seconds
    #     of feature data
    # Output:
    #   * List of log_PSD tuples
    def log_PSDs(self, start = None, end = None, sample_window = 512):
        log_PSDs = np.array([])
        while start + sample_window < end:
            np.append(log_PSDs, self.log_PSD(start, start+sample_window))
            start += sample_window
        return log_PSDs


    # Calculate the heart rate number from number of peaks and start:end timeframe 
    # Input:
    #   * start_time (datetime), end_time (datetime)
    # Output:
    #   * heart rate in beats per minute (float)
    def get_BPM_and_peak_variance(self, start = None, end = None):

        # Focus on standard HR frequency range
        segment = self.bandpass_filter(0.8, 2.5, start, end)

        # Find the segment peaks' indices for peak occurrence variance feature.
        indices = detect_peaks.detect_peaks(segment, mph = 50)

        peak_variance = np.finfo(np.float32).max
        if len(indices) > 1:
            peak_variance = np.var(np.diff(indices))

        time_difference = self.timestamp_in_datetime(end) - self.timestamp_in_datetime(start)
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
