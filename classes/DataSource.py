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
import os.path
import itertools
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.regularizers import l2, activity_l2
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping, TensorBoard
from keras.utils import np_utils
import matplotlib.pyplot as plt

class DataSource:

	def __init__(self):
		self.positive_features = []
		self.negative_features = []
		self.used_ranges = []
		self.dataset = None

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

	def load_entire_dataset(self):
		features = pd.DataFrame({"feature_vec", "file_id", "start", "end"})
		for file_id in range(20):
			data = self.read_data_from_file(file_id)
	        start = 0
	        step  = 256
	        while start+step < data.shape[0]:
	        	if [file_id, start, start+step] in self.used_ranges:
	        		continue
	            segment = data.iloc[start:start+step]
	            signal = Signal(segment["ppg"].values, segment["timestamp"].values)
	            features.append(signal.extract_PSD_features())
	            features.append({"feature_vec": signal.extract_PSD_features(),
	            				 "file_id"    : file_id,
	            				 "start"      : start,
	            				 "end"        : start+step})
	            start += step

        feature_vecs = np.array(features["feature_vec"].values)
        feature_vecs = self.standardize_and_reduce_dim(feature_vecs, "retrain")
        features["feature_vec"] = feature_vecs
        self.dataset = features

	# This is called in the beginning only
	def load_labeled_data(self, num_records=50):
        features = []
        labels = []
        for range_type in range(0,2):
            if range_type == 0:
                print("Reading non-HR segments")
                range_file = "data/non_HR_ranges.csv"
            else:
                print("Reading HR segments")
                range_file = "data/HR_ranges.csv"

            ranges = pd.read_csv(
                range_file, 
                header=None, 
                names=["file_id", "start", "end"],
                nrows=num_records)

            gb = ranges.groupby(["file_id"])

            for file_id, indices in gb.groups.items():
                data = self.read_data_from_file(file_id)
                for i in indices:
                    range_start = ranges.loc[i,"start"]
                    range_end = ranges.loc[i,"end"]
                    segment = data.iloc[range_start:range_end]
                    signal = Signal(segment["ppg"].values, segment["timestamp"].values)
                    features.append(signal.extract_PSD_features())
                    labels.append(range_type)
                    self.used_ranges.append([file_id, range_start, range_end])

        features = np.array(features)
        features = self.standardize_and_reduce_dim(features, "train")
        
        # One hot encode labels
        labels = np.array(labels)
        labels = np_utils.to_categorical(labels, 2)
        
        return features, labels

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
        features = self.standardize_and_reduce_dim(features, "retrain")

        if include_signals:
            return features, signals
        return features

	def standardize_and_reduce_dim(self, features, dataset_type="train"):
        if self.scaler == None:
            if dataset_type == "train":
                scaler = preprocessing.StandardScaler().fit(features)
                np.save("scaler", scaler)
            else:
                scaler = np.load("scaler")
            self.scaler = scaler
        features = self.scaler.transform(features)

        if self.pca == None:
            if dataset_type == "train":
                # Reduce dimensionality
                pca = PCA(n_components=30)
                pca = pca.fit(features)
                np.save("pca", pca)
            else:
                pca = np.load("pca")
            self.pca = pca

        features = self.pca.transform(features)
        return features