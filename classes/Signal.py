
import numpy as np
import scipy

class Signal:

    def __init__(self, signal, fs = 80):
        self.content = np.array(signal)
        self.sample_freq = fs

    # Correct signal saturation (defined as sharp slope in curve) 
    # by scaling subsequent readings by the maximum range
    # Params, Return: list of PPG readings (floats)
    def correctSaturation(self):
        max_val = max(self.content)
        signal_diff = np.diff(self.content)
        signal_diff_end = len(signal_diff)-1
        max_slope = 50000

        for k in range(len(signal_diff)):
            if signal_diff[k] > max_slope:
                # Pull subsequent values down
                self.content[k+1:signal_diff_end] = self.content[k+1:signal_diff_end] - max_val
            if signal_diff[k] < -1*max_slope:
                # Pull subsequent values up
                self.content[k+1:signal_diff_end] = self.content[k+1:signal_diff_end] + max_val

    # Filter outliers using a moving median filter
    def removeOutliers(self, window = 3):
        self.content = scipy.signal.medfilt(self.content, window)

    # Log power spectral density of the signal segment (used as a feature)
    # Input:
    #   * start, end: indeces of signal segment range
    #   * sample_window: length of overlapping segments for pwelch averaging
    # Output:
    #   Scaled log base 10 of the PSD corresponding to the segment start-end
    def logPSD(self, start = None, end = None, sample_window = 128):
        f, psd = scipy.signal.welch(self.content, fs = self.sample_freq, nperseg = sample_window)
        return 10*np.log10(psd)

    # List of log PSDs of the sample_window-sized segments in the signal
    # between start and end.
    # Input:
    #   * start, end: indeces of signal segment range
    #   * sample_window: max-length of each segment, default to 512 ~6 seconds
    #     of feature data
    def logPSDs(self, start = None, end = None, sample_window = 512)
        logPSDs = []
        while start + sample_window < end:
            logPSDs.append(self.logPSD(start, start+sample_window))
            start += sample_window
        return logPSDs

    def getPeaks(self):
        # Apply a 2nd order bandpass butterworth filter to attenuate heart rate components
        # other than heart signal's RR peaks.
        b, a = signal.butter(2, [1/100, 1/10], 'bandpass')
        dataWindow = signal.filtfilt(b, a, dataWindow)
        # Find the segment peaks' indices for peak occurrence variance feature.
        indices = detect_peaks.detect_peaks(dataWindow, mph = 30, mpd = 20)
        peakVariance = np.finfo(np.float64).max
        if len(indices) > 1:
            peakVariance = np.var(np.diff(indices))
