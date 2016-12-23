import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, svm
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.externals import joblib
import scipy
import pandas as pd
from classes.Signal import Signal
import os
import itertools
import matplotlib.pyplot as plt

class DataSource:

    dataset = None
    scaler = None
    pca = None
    figure_id = 1

    def __init__(self, num_labeled_records=256, pca_n_components=30):
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
        if os.path.isfile("dataset.pkl"):
            self.dataset = pd.read_pickle("dataset.pkl")
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

            feature_vecs = np.vstack(dataset["feature_vec"])
            feature_vecs = self.standardize_and_reduce_dim(feature_vecs)
            dataset["feature_vec"] = list(feature_vecs)

            dataset.reset_index(drop=True, inplace=True)
            dataset.to_pickle("dataset.pkl")
            self.dataset = dataset

    # Called once in the beginning only and during tests
    def load_or_process_labeled_dataset(self, from_file_id=None):
        if from_file_id==None and os.path.isfile("dataset_labeled.pkl"):
            return pd.read_pickle("dataset_labeled.pkl")
        else:
            columns = ["feature_vec", "file_id", "start", "end", "label"]
            if from_file_id:
                columns.append("signal")
            dataset = pd.DataFrame(columns=columns)
            features = []
            labels = []
            for range_type in range(0,2):
                if range_type == 0:
                    print("Reading non-HR segments")
                    range_file = "data/negative_ranges.csv"
                else:
                    print("Reading HR segments")
                    range_file = "data/positive_ranges.csv"

                ranges = pd.read_csv(
                        range_file, 
                        header=None, 
                        names=["file_id", "start", "end"])
                if from_file_id:
                    ranges = ranges[ranges["file_id"] == from_file_id]
                else:
                    # Reject test set samples
                    ranges = ranges[ranges["file_id"] != 20.0]
                    # Take {num_labeled_records} random samples
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

            feature_vecs = np.vstack(dataset["feature_vec"])
            labels = None
            if from_file_id == None:
                labels = dataset["label"].values
            feature_vecs = self.standardize_and_reduce_dim(feature_vecs, labels)
            dataset["feature_vec"] = list(feature_vecs)
            dataset.reset_index(drop=True, inplace=True)
            dataset = dataset.reindex(np.random.permutation(dataset.index))

            if from_file_id==None:
                dataset.to_pickle("dataset_labeled.pkl")
            return dataset

    def standardize_and_reduce_dim(self, features, labels=None):
        if self.pca == None:
            if os.path.isfile("pca.pkl") == False:
                pca = PCA(n_components=self.pca_n_components, whiten=True)
                pca = pca.fit(features, labels)
                joblib.dump(pca, 'pca.pkl') 
            else:
                print("load PCA params")
                pca = joblib.load('pca.pkl') 
            self.pca = pca
        features = self.pca.transform(features)

        if self.scaler == None:
            if os.path.isfile("scaler.pkl") == False:
                scaler = preprocessing.StandardScaler().fit(features)
                joblib.dump(scaler, 'scaler.pkl') 
            else:
                print("load scaler")
                scaler = joblib.load("scaler.pkl")
            self.scaler = scaler
        features = self.scaler.transform(features)

        return features

    def display_dataset(self, dataset):
        num_figures = 3
        num_figure_subplots = 10
        for k in range(num_figures):
            print("figure id:", self.figure_id)
            plt.figure(self.figure_id)
            for i in range(num_figure_subplots):
                e = dataset.iloc[k*num_figure_subplots + i]

                # remove this - just for testing
                if "signal" not in e:
                    dd = self.read_data_from_file(e.file_id)
                    segment = dd.iloc[int(e.start):int(e.end)]
                    e.signal = Signal(segment["ppg"].values, segment["timestamp"].values)
                
                signal = e.signal

                signal_filtered = signal.bandpass_filter(0.8, 2.5)

                signal_scaled = preprocessing.scale(signal.bandpass_filter(0.1, 20))
                signal_filtered_scaled = preprocessing.scale(signal_filtered)

                ax = plt.subplot(10,2,2*i+1)

                #remove this - just for testing
                if "label" in e:
                    printout = e.label
                else:
                    printout = e.pred
                ax.text(0.5, 0.5, printout,
                horizontalalignment='center',
                verticalalignment='top',
                transform=ax.transAxes,
                fontsize=12)

                ax.plot(signal_scaled)
                ax.plot(signal_filtered_scaled, color='r')

                peaks = scipy.signal.find_peaks_cwt(signal_filtered_scaled, np.arange(5,10))
                ax.plot(peaks, signal_filtered_scaled[peaks], marker='o', linestyle="None")

                ax = plt.subplot(10,2,2*i+2)
                ax.plot(e.feature_vec)
                print("%d,%d,%d" %(int(e.file_id), int(e.start), int(e.end)))

            plt.show()
            #plt.savefig('plots/plot%d.png' % self.figure_id)
            self.figure_id += 1