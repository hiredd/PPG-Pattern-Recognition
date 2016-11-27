# Take one data file, extract 512-long segments from it,
# for each segment, extract the 3 features, then sort the
# resulting feature array from the whole file by those 3 features
# using a minimizing overall function. plot the signal for the
# times that correspond to the maximum features

import sys
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter, freqz, filtfilt, welch, medfilt
from datetime import datetime
sys.path.append('lib')
import detect_peaks
from sklearn import preprocessing
import matplotlib.pyplot as plt
from peakutils.peak import indexes

def correct_saturation(content):
    max_val = max(content)
    signal_diff = np.diff(content)
    signal_diff_end = len(signal_diff)-1
    max_slope = 50000

    for k in range(len(signal_diff)):
        if signal_diff[k] > max_slope:
            # Pull subsequent values down
            content[k+1:signal_diff_end] = content[k+1:signal_diff_end] - max_val
        if signal_diff[k] < -1*max_slope:
            # Pull subsequent values up
            content[k+1:signal_diff_end] = content[k+1:signal_diff_end] + max_val
    return content

def remove_outliers(content, window = 3):
    return medfilt(content, window)

def butter_bandpass_filter(data, cutoff_low, cutoff_high, fs, order=2):
    nyq = 0.5 * fs
    normal_cutoff_low = cutoff_low / nyq
    normal_cutoff_high = cutoff_high / nyq
    b, a = butter(order, [normal_cutoff_low, normal_cutoff_high], btype='bandpass', analog=False)

    y = filtfilt(b, a, data)
    return y

def i_freq(lf, frequency):
        i = 0
        while lf[i] < frequency:
            i += 1
        if i == 0:
            return 0
        return i-1

def get_BPM_and_peak_variance(signal, time_start, time_end):
    signal = butter_bandpass_filter(signal, 0.8, 2.5, 80)
    indices = detect_peaks.detect_peaks(signal, mph = 50)
    #indices = indexes(np.array(signal), thres = 50/max(signal))

    peak_variance = np.finfo(np.float32).max
    if len(indices) > 1:
        peak_variance = np.var(np.diff(indices))

    time_difference = time_end-time_start
    time_difference_in_minutes = float(time_difference.seconds + 
                                 float(time_difference.microseconds)/10**6)/60.0
    BPM = float(len(indices)) / time_difference_in_minutes
    return BPM, peak_variance

def extract_features(signal, time_start, time_end):

    #signal = butter_bandpass_filter(signal, 0.8, 2.5, 80)

    feature_vector = []

    f, psd = welch(signal, fs = 80, nperseg = 128)
    psd = 10*np.log10(psd)

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
    BPM, peak_variance = get_BPM_and_peak_variance(signal, time_start, time_end)
    feature_vector.append(peak_variance)

    if BPM < 50 or BPM > 120:
        return None

    return feature_vector

def string_to_datetime(s):
    try:
        timestamp = datetime.strptime(s, '%Y-%m-%d %H:%M:%S.%f')
    except ValueError:
        timestamp = datetime.strptime(s, '%Y-%m-%d %H:%M:%S')
    return timestamp

current_file_id = int(sys.argv[1])

df = pd.read_csv("data/data%d.csv" % current_file_id, header = 0, index_col = 0)
df.loc[:,"ppg"] = remove_outliers(correct_saturation(df.loc[:,"ppg"].values))

step = 512/2
start = 0
end = np.shape(df)[0]
features = []
while start+step < end:
    segment = df.loc[start:start+step,:]

    start_time = string_to_datetime(segment.head(1).timestamp.values[0])
    end_time = string_to_datetime(segment.tail(1).timestamp.values[0])
    start_id = start
    end_id = start+step

    feature_vector = extract_features(segment.loc[:,"ppg"].values, start_time, end_time)
    if feature_vector != None:
        feature_vector += [start_time, end_time, start_id, end_id]
        features.append(feature_vector)

    start += step

features = pd.DataFrame(features)

# Sort by features in ascending order
features = features.sort_values([2,1,3,0])

def onclick(event):
    fx, fy = fig.transFigure.inverted().transform((event.x,event.y))

    for i, subplot in enumerate(subplots):
        if subplot["pos"].contains(fx,fy) and subplot["used"] == False:
            range_ids = pd.DataFrame([subplot["range"]])
            range_ids.to_csv('data/HR_ranges.csv', mode='a', header=False, index=False)
            subplots[i]["used"] = True
            break

num_figure_subplots = 30
counter = 0
k = 0
while num_figure_subplots*k < features.shape[0]:
    fig = plt.figure(k+1)
    subplots = []
    for i in range(num_figure_subplots):
        feat = features.iloc[num_figure_subplots*k+i]

        ax = plt.subplot(num_figure_subplots/3,3,i+1)
        #plt.title("file: %s, start: %d, end: %d" % ("data15", feat[6], feat[7]))

        start = pd.Timestamp(feat[4])
        end = pd.Timestamp(feat[5])
        t = np.linspace(start.value, end.value, step+1)
        t = pd.to_datetime(t)
        plt.plot(t, df.loc[feat[6]:feat[7],"ppg"])
        subplots.append({"pos":ax.get_position(), "range":[current_file_id, int(feat[6]), int(feat[7])], "used":False})
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()

    counter += num_figure_subplots
    k += 1