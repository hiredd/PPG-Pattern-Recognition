import numpy as np
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, svm
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
import scipy
import tensorflow as tf
import pandas as pd
from classes.Signal import Signal
import os
import itertools
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.regularizers import l2, activity_l2
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping, TensorBoard
from keras.utils import np_utils
import matplotlib.pyplot as plt

class DataSource:

    dataset = None
    scaler = None
    pca = None

    def __init__(self, num_labeled_records=100, pca_n_components=30):
        self.num_labeled_records=num_labeled_records
        self.pca_n_components = pca_n_components

    def read_data_from_file(self, file_id):
        data = pd.read_csv(
                "data/data%d.csv" % file_id, 
                header=0, 
                index_col=0, 
                dtype = {
                    "device": np.int32,
                    "ppg":    np.float32, 
                    "accx":   np.float32, 
                    "accy":   np.float32, 
                    "accz":   np.float32})
        return data

    def load_or_process_entire_dataset(self):
        if os.path.isfile("dataset.h5"):
            self.dataset = pd.read_hdf("dataset.h5")
        else:
            columns = ["feature_vec", "file_id", "start", "end", "pred"]
            dataset = pd.DataFrame(columns=columns)
            for file_id in range(0,20):
                data = self.read_data_from_file(file_id)
                print("Processing file data%d.csv" % file_id, end='')
                start = 0
                step  = 256
                data_size = data.shape[0]
                while start+step < data_size:
                    if start % (data_size//10) < step:
                        print(".", sep=' ', end='', flush=True)
                    segment = data.iloc[start:start+step]
                    signal = Signal(segment["ppg"].values, segment["timestamp"].values)
                    dataset = dataset.append(pd.DataFrame([[signal.extract_PSD_features(),
                                     file_id,
                                     start,
                                     start+step,
                                     0]],columns=columns))
                    start += step
                print()
                print("Total feature vectors generated so far:", dataset.shape[0])

            feature_vecs = dataset["feature_vec"].values
            feature_vecs = np.array([v for v in feature_vecs])
            feature_vecs = self.standardize_and_reduce_dim(feature_vecs)
            dataset["feature_vec"] = list(feature_vecs)

            dataset = dataset.reset_index()
            dataset.to_hdf("dataset.h5", "dataset")
            self.dataset = dataset

    # Called once in the beginning only and during tests
    def load_labeled_dataset(self, from_file_id=None):
        columns = ["feature_vec", "file_id", "start", "end", "label"]
        if from_file_id:
            columns.append("signal")
        dataset = pd.DataFrame(columns=columns)
        features = []
        labels = []
        for range_type in range(0,2):
            if range_type == 0:
                print("Reading non-HR segments")
                range_file = "data/non_HR_ranges.csv"
            else:
                print("Reading HR segments")
                range_file = "data/HR_ranges.csv"

            if from_file_id:
                ranges = pd.read_csv(
                    range_file, 
                    header=None, 
                    names=["file_id", "start", "end"])
                ranges = ranges[ranges["file_id"] == from_file_id]
            else:
                ranges = pd.read_csv(
                    range_file, 
                    header=None, 
                    names=["file_id", "start", "end"])
                ranges = ranges.reindex(np.random.permutation(ranges.index))
                ranges = ranges.iloc[:self.num_labeled_records]

            gb = ranges.groupby(["file_id"])

            for file_id, indices in gb.groups.items():
                data = self.read_data_from_file(file_id)
                for i in indices:
                    start = ranges.loc[i,"start"]
                    end = ranges.loc[i,"end"]
                    segment = data.iloc[start:end]
                    signal = Signal(segment["ppg"].values, segment["timestamp"].values)

                    dataset_entry = [signal.extract_PSD_features(),
                                     file_id,
                                     start,
                                     end,
                                     range_type]
                    if from_file_id:
                        dataset_entry.append(signal)

                    dataset = dataset.append(pd.DataFrame([dataset_entry],columns=columns))

        feature_vecs = dataset["feature_vec"].values
        feature_vecs = np.array([v for v in feature_vecs])
        feature_vecs = self.standardize_and_reduce_dim(feature_vecs, init=True)
        dataset["feature_vec"] = list(feature_vecs)
        return dataset

    def load_unlabeled_data_from_file(self, file_id, include_signals=False):
        data = self.read_data_from_file(file_id)
        start = 0
        step  = 256
        features = []
        signals = []
        while start+step < data.shape[0]:
            segment = data.iloc[start:start+step]
            signal = Signal(segment["ppg"].values, segment["timestamp"].values)
            signals.append(signal)
            features.append(signal.extract_PSD_features())
            start += step

        features = np.array(features)
        features = self.standardize_and_reduce_dim(features)

        if include_signals:
            return features, signals
        return features

    def standardize_and_reduce_dim(self, features, init=False):
        if self.scaler == None:
            if init:
                scaler = preprocessing.StandardScaler().fit(features)
                np.save("scaler", scaler)
            else:
                scaler = np.load("scaler")
            self.scaler = scaler
        features = self.scaler.transform(features)

        if self.pca == None:
            if init:
                # Reduce dimensionality
                pca = PCA(n_components=self.pca_n_components)
                pca = pca.fit(features)
                np.save("pca", pca)
            else:
                pca = np.load("pca")
            self.pca = pca

        features = self.pca.transform(features)
        return features

    def display_dataset(self, dataset):
        num_figures = 3
        num_figure_subplots = 30
        for k in range(num_figures):
            plt.figure(k+1)
            for i in range(num_figure_subplots):
                e = dataset.iloc[k*num_figure_subplots + i]
                signal = e.signal

                signal_filtered = signal.bandpass_filter(0.8, 2.5)

                signal_scaled = preprocessing.scale(signal.content)
                signal_filtered_scaled = preprocessing.scale(signal_filtered)

                plt.subplot(num_figure_subplots/3,3,i+1)
                plt.plot(signal_scaled)
                plt.plot(signal_filtered_scaled, color='r')

                peaks = scipy.signal.find_peaks_cwt(signal_filtered_scaled, np.arange(5,10))
                plt.plot(peaks, signal_filtered_scaled[peaks], marker='o', linestyle="None")
            plt.savefig('plots/plot%d.png' % k)