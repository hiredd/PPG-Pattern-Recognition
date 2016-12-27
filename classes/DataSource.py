import numpy as np
import scipy
import pandas as pd
import os
import itertools
import argparse
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from helpers import *
from classes.Signal import Signal

class DataSource:

    test_set_file = 20
    timestep_size = 256
    dataset = None
    scaler = None
    pca = None
    figure_id = 1

    def __init__(self, num_labeled_records=500, pca_n_components=10):
        self.num_labeled_records=num_labeled_records
        self.pca_n_components = pca_n_components

        parser = argparse.ArgumentParser()
        parser.add_argument('--regenerate', action='store_true')
        args = parser.parse_args()
        self.should_regenerate_feature_files = args.regenerate

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
        if os.path.isfile("cache/dataset.pkl") and self.should_regenerate_feature_files==False:
            self.dataset = pd.read_pickle("cache/dataset.pkl")
        else:
            columns = ["feature_vec", "file_id", "start", "end", "pred"]
            dataset = pd.DataFrame(columns=columns)
            for file_id in range(0,21):
                if file_id == self.test_set_file:
                    continue
                data = self.read_data_from_file(file_id)
                print("Processing file data%d.csv" % file_id, end='')
                start = 0
                step  = self.timestep_size
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
                print("\nTotal feature vectors generated so far:", dataset.shape[0])

            feature_vecs = np.vstack(dataset["feature_vec"])
            feature_vecs = self.standardize_and_reduce_dim(feature_vecs)
            dataset["feature_vec"] = list(feature_vecs)

            dataset.reset_index(drop=True, inplace=True)
            dataset.to_pickle("cache/dataset.pkl")
            self.dataset = dataset

    def load_or_process_labeled_dataset(self, from_file_id=None):
        if from_file_id==None and \
           os.path.isfile("cache/dataset_labeled.pkl") and \
           self.should_regenerate_feature_files==False:
            return pd.read_pickle("cache/dataset_labeled.pkl")
        else:
            columns = ["feature_vec", "file_id", "start", "end", "label"]
            if from_file_id:
                columns.append("signal")
            dataset = pd.DataFrame(columns=columns)
            features = []
            labels = []
            for range_type in range(0,2):
                if range_type == 0:
                    print("Reading negative-labeled segments")
                    range_file = "data/negative_ranges.csv"
                else:
                    print("Reading positive-labeled segments")
                    range_file = "data/positive_ranges.csv"

                ranges = pd.read_csv(
                        range_file, 
                        header=None, 
                        names=["file_id", "start", "end"])
                if from_file_id:
                    ranges = ranges[ranges["file_id"] == from_file_id]
                else:
                    # Reject test set samples
                    ranges = ranges[ranges["file_id"] != self.test_set_file + 0.0]
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
                dataset.to_pickle("cache/dataset_labeled.pkl")
            return dataset

    def standardize_and_reduce_dim(self, features, labels=None):
        if self.pca == None:
            if os.path.isfile("cache/pca.pkl") == False or self.should_regenerate_feature_files:
                pca = PCA(n_components=self.pca_n_components, whiten=True)
                pca = pca.fit(features, labels)
                joblib.dump(pca, "cache/pca.pkl") 
            else:
                print("load PCA params")
                pca = joblib.load("cache/pca.pkl") 
            self.pca = pca
        features = self.pca.transform(features)

        if self.scaler == None:
            if os.path.isfile("cache/scaler.pkl") == False or self.should_regenerate_feature_files:
                scaler = preprocessing.StandardScaler().fit(features)
                joblib.dump(scaler, "cache/scaler.pkl") 
            else:
                print("load scaler")
                scaler = joblib.load("cache/scaler.pkl")
            self.scaler = scaler
        features = self.scaler.transform(features)

        return features

    def display_dataset(self, dataset):
        num_figures = 3
        num_figure_subplots = 2
        for k in range(num_figures):
            plt.figure(self.figure_id)
            for i in range(num_figure_subplots):
                e = dataset.iloc[k*num_figure_subplots + i]
                if i==1:
                    e = dataset.iloc[-200]
                else:
                    e = dataset.iloc[100]

                if "signal" not in e:
                    dd = self.read_data_from_file(e.file_id)
                    segment = dd.iloc[int(e.start):int(e.end)]
                    e.signal = Signal(segment["ppg"].values, segment["timestamp"].values)
                
                signal = preprocessing.scale(e.signal.highpass_filter(1))
                signal_filtered = preprocessing.scale(e.signal.bandpass_filter(0.8, 2.5))

                start_time = pd.Timestamp(e.signal.timestamp_in_datetime(0))
                end_time = pd.Timestamp(e.signal.timestamp_in_datetime(-1))
                t = np.linspace(start_time.value, end_time.value, self.timestep_size)
                t = pd.to_datetime(t)

                ax = plt.subplot(num_figure_subplots,1,i+1)

                # Display the true label or prediction on the plot
                if False:
                    if "label" in e:
                        printout = e.label
                    else:
                        printout = e.pred
                    ax.text(0.5, 0.5, printout,
                        horizontalalignment='center',
                        verticalalignment='top',
                        transform=ax.transAxes,
                        fontsize=12)

                ax.plot(t, signal)
                ax.plot(t, signal_filtered, color='r')
                ax.xaxis_date()
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                ax.yaxis.set_visible(False)

            plt.show()
            #plt.savefig('plots/plot%d.png' % self.figure_id)
            self.figure_id += 1

    def confusion(self, dataset):
        preds = np.rint(dataset.pred.values)
        labels = dataset.label
        cnf_matrix = confusion_matrix(labels, preds)
        np.set_printoptions(precision=2)
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=["Positive","Negative"], title='')

    def remove_labeled_subset_from_dataset(self, labeled_ds):
        for i in range(labeled_ds.shape[0]):
            file_id = labeled_ds.iloc[i]["file_id"]
            start = labeled_ds.iloc[i]["start"] 
            end = labeled_ds.iloc[i]["end"]
            self.dataset = self.dataset[~((self.dataset.file_id == file_id) & 
                                    (self.dataset.start == start) & 
                                    (self.dataset.end == end))]
        self.dataset.reset_index(drop=True, inplace=True)
