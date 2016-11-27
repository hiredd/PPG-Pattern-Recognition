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
from classes.Signal import Signal

def standardize(signal):
    return preprocessing.scale(signal)

def onclick(event):
    fx, fy = fig.transFigure.inverted().transform((event.x,event.y))

    for i, subplot in enumerate(subplots):
        if subplot["pos"].contains(fx,fy) and subplot["used"] == False:
            range_ids = pd.DataFrame([subplot["range"]])
            range_ids.to_csv('data/HR_ranges.csv', mode='a', header=False, index=False)
            subplots[i]["used"] = True
            break

current_file_id = int(sys.argv[1])

df = pd.read_csv("data/data%d.csv" % current_file_id, 
        header = 0, 
        index_col = 0, 
        dtype = {"device":np.int32,
                 "ppg":np.float32, 
                 "accx":np.float32, 
                 "accy":np.float32, 
                 "accz": np.float32})
signal = Signal(df.loc[:,"ppg"].values, df.loc[:,"timestamp"].values)
signal.correct_saturation()
signal.remove_outliers()

step = 256
start, end = 0, signal.length()
features = []
while start+step < end:
    feature_vector = signal.extract_features(start, start+step)
    
    if feature_vector != None:
        features.append(feature_vector + [start,start+step])

    start += step

# Sort by features in ascending order, in order of feature importance
features = pd.DataFrame(features).sort_values([2,1,3,0])

df = pd.read_csv("data/HR_ranges.csv")

review = True
label_non_HR = True

if label_non_HR:
    for i in range(1000):
        if i >= features.shape[0]:
            break
        feat = features.iloc[i]
        start, end = int(feat[4]), int(feat[5])
        if df.isin([current_file_id, start, end]).all(1).any() == False:
            range_ids = pd.DataFrame([[current_file_id, start, end]])
            range_ids.to_csv('data/non_HR_ranges.csv', mode='a', header=False, index=False)
    quit()

num_figure_subplots = 30
counter = 0
k = 0
while num_figure_subplots*k < features.shape[0] and k < 100:
    fig = plt.figure(k+1)
    subplots = []
    for i in range(num_figure_subplots):
        feat = features.iloc[num_figure_subplots*k+i]
        start, end = int(feat[4]), int(feat[5])

        signal_filtered = signal.bandpass_filter(0.8, 2.5, start, end)

        start_time = pd.Timestamp(signal.timestamp_in_datetime(start))
        end_time = pd.Timestamp(signal.timestamp_in_datetime(end))
        t = np.linspace(start_time.value, end_time.value, step)
        t = pd.to_datetime(t)

        ax = plt.subplot(num_figure_subplots/3,3,i+1)
        subplots.append({"pos":ax.get_position(), 
                         "range":[current_file_id, start, end], 
                         "used":False})

        if review:
            if df.isin([current_file_id, start, end]).all(1).any() == False:
                plt.plot(t, standardize(signal.content[start:end]))
                plt.plot(t, standardize(signal_filtered), color='r')
        else:
            plt.plot(t, standardize(signal.content[start:end]))
            plt.plot(t, standardize(signal_filtered), color='r')

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()

    counter += num_figure_subplots
    k += 1
