import numpy as np
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, svm
from sklearn.ensemble import BaggingClassifier
import tensorflow as tf
import pandas as pd
from classes.Signal import Signal
import os.path
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam, SGD
from keras.utils import np_utils
from keras.models import load_model
import matplotlib.pyplot as plt

class HRClassifier:

    def __init__(self, NN = True, only_psds = True):
        self.features = []
        self.labels = []
        self.only_psds = only_psds
        self.nn = NN

        self.model = None
        if NN:
            self.model_file_name = 'trained_model.h5'
            if os.path.exists(self.model_file_name):
                self.model = load_model(self.model_file_name)
        else:
            self.model_file_name = 'trained_model.pkl'
            if os.path.exists(self.model_file_name):
                self.model = joblib.load(self.model_file_name) 


    def load_data(self):
        
        num_data_files = 21

        for range_type in range(2):

            if range_type == 0:
                print("Reading non-HR segments from files:")
                range_file = "data/non_HR_ranges.csv"
            else:
                print("Reading HR segments from files:")
                range_file = "data/HR_ranges.csv"

            ranges = pd.read_csv(
                range_file, 
                header=None, 
                names=["file_id", "start", "end"])

            for file_id in range(num_data_files):
                print(file_id)

                if ranges[ranges["file_id"] == file_id].size > 0:

                    data = pd.read_csv(
                        "data/data%d.csv" % file_id, 
                        header=0, 
                        index_col=0, 
                        dtype = {
                            "device":np.int32,
                            "ppg":np.float32, 
                            "accx":np.float32, 
                            "accy":np.float32, 
                            "accz": np.float32})

                    for _,row in ranges[ranges["file_id"] == file_id].iterrows():
                        segment = data.iloc[row["start"]:row["end"]]
                        s = Signal(segment["ppg"].values, segment["timestamp"].values)

                        if self.only_psds:
                            f, psd = s.log_PSD()
                            self.features.append(psd)
                        else:
                            self.features.append(s.extract_features(None, None, False))

                        self.labels.append(range_type)

        self.features = np.array(self.features)
        self.features = preprocessing.scale(self.features)
        self.labels = np.array(self.labels)


    def train(self, train_anyway = False):
        if not train_anyway:
            assert self.model == None, "A trained model already exists under %s." % self.model_file_name

        self.load_data()

        train_x, test_x, train_y, test_y = train_test_split(self.features, 
                                                            self.labels, 
                                                            test_size=0.2)

        if self.nn == False:
            print(train_x)
            model = svm.SVC(verbose = True)
            model.fit(train_x, train_y) 
            self.model = model
            joblib.dump(model, self.model_file_name) 

            print(model.score(test_x, test_y))
            return

        train_y = np_utils.to_categorical(train_y, 2)
        test_y = np_utils.to_categorical(test_y, 2)

        # Hyperparameters
        learning_rate = 10**(-3)
        num_epochs = 20
        batch_size = 32
        n_inputs, n_input_features = np.shape(train_x)

        if self.only_psds:
            l1_dim = 128
            l2_dim = 32
        else:
            l1_dim = 8
            l2_dim = 4

        model = Sequential()
        model.add(Dense(128, input_dim=train_x.shape[1], activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(2, activation="softmax"))

        optimizer = Adam(lr=learning_rate)
        model.compile(loss='categorical_crossentropy', 
            metrics=['accuracy'], 
            optimizer=optimizer)

        model.fit(train_x, 
            train_y,
            batch_size=batch_size, 
            nb_epoch=num_epochs, 
            shuffle=True)
        
        scores = model.evaluate(test_x, test_y, verbose=0, batch_size=batch_size)
        print("Accuracy: %.2f%%" % (scores[1]*100))

        self.model = model
        self.model.save(self.model_file_name)


    def validate(self, file_id):

        assert self.nn == True, "Validate method currently only supported for NN implementation."

        data = pd.read_csv("data/data%d.csv" % file_id, 
            header=0, 
            index_col=0, 
            dtype = {
                "device":np.int32,
                "ppg":np.float32, 
                "accx":np.float32, 
                "accy":np.float32, 
                "accz": np.float32})

        ranges = []
        signal = Signal(data["ppg"].values, data["timestamp"].values)
        start = 0
        end = data.shape[0]
        step = 256
        while start+step < end:
        
            if self.only_psds:
                f, psd = signal.log_PSD(start, start+step)
                feat_vec = psd
            else:
                feat_vec = signal.extract_features(start, start+step, False)

            batch_size = 32
            acc = self.model.predict_proba(np.array([feat_vec]), batch_size=batch_size)
            acc = acc[0][1]

            ranges.append([acc, (start, start+step)])

            start += step 

        df = pd.DataFrame(ranges)

        df = df.sort_values(0)

        num_figure_subplots = 30
        plt.figure(1)
        for i in range(num_figure_subplots):
            start, end = df.iloc[i][1]

            signal_filtered = signal.bandpass_filter(0.8, 2.5, start, end)

            plt.subplot(num_figure_subplots/3,3,i+1)
            plt.plot(preprocessing.scale(signal.content[start:end]))
            plt.plot(preprocessing.scale(signal_filtered), color='r')
        plt.show()