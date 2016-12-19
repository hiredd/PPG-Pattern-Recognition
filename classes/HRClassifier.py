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

class HRClassifier:

    def __init__(self, use_nn=True):
        self.use_nn = use_nn
        self.scaler = None
        self.pca = None
        self.plot_id = 0

        if use_nn:
            self.model_file_name = 'models/trained_model.h5'
            if os.path.exists(self.model_file_name):
                self.model = load_model(self.model_file_name)
        else:
            self.model_file_name = 'models/trained_model.pkl'
            if os.path.exists(self.model_file_name):
                self.model = joblib.load(self.model_file_name) 


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


    def load_unlabeled_data_from_file(self, file_id, include_signals=False):
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

    def load_labeled_data(self, num_records=50):
        features = []
        labels = []
        for range_type in range(0,2):
            if range_type == 0:
                print("Reading non-HR segments from file(s):")
                range_file = "data/non_HR_ranges.csv"
            else:
                print("Reading HR segments from file(s):")
                range_file = "data/HR_ranges.csv"

            ranges = pd.read_csv(
                range_file, 
                header=None, 
                names=["file_id", "start", "end"],
                nrows=num_records)

            gb = ranges.groupby(["file_id"])

            for file_id, indices in gb.groups.items():
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
                for i in indices:
                    print(i)
                    range_start = ranges.loc[i,"start"]
                    range_end = ranges.loc[i,"end"]
                    segment = data.iloc[range_start:range_end]
                    signal = Signal(segment["ppg"].values, segment["timestamp"].values)
                    features.append(signal.extract_PSD_features())
                    labels.append(range_type)

        features = np.array(features)
        features = self.standardize_and_reduce_dim(features, "train")
        
        labels = np.array(labels)
        labels = np_utils.to_categorical(labels, 2)
        
        return features, labels

    def load_all_labeled_data(self, dataset_type="train", file_ids=np.asarray(range(1,10))):
        features = []
        labels = []

        for range_type in range(2):
            if range_type == 0:
                print("Reading non-HR segments from file(s):")
                range_file = "data/non_HR_ranges.csv"
            else:
                print("Reading HR segments from file(s):")
                range_file = "data/HR_ranges.csv"

            ranges = pd.read_csv(
                range_file, 
                header=None, 
                names=["file_id", "start", "end"])

            for file_id in file_ids:
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

                        signal = Signal(segment["ppg"].values, segment["timestamp"].values)

                        # Add more positively labeled examples by horizontally flipping
                        # the signal. The PSD stays the same, but technically different signal.
                        #signals += [Signal(segment["ppg"].iloc[::-1], segment["timestamp"].values)]
                        features.append(signal.extract_PSD_features())
                        labels.append(range_type)

        features = np.array(features)
        features = self.standardize_and_reduce_dim(features, dataset_type)
        
        labels = np.array(labels)
        labels = np_utils.to_categorical(labels, 2)
        
        return features, labels

    def init_MLP_model(self, input_dim):
        model = Sequential()
        model.add(Dense(128, input_dim=input_dim, activation="relu"))
        model.add(Dropout(0.2))
        #model.add(Dense(32, activation="relu"))
        #model.add(Dropout(0.2))
        model.add(Dense(2, activation="softmax"))

        learning_rate = 10**(-3)
        optimizer = Adam(lr=learning_rate)
        model.compile(loss='categorical_crossentropy', 
            metrics=['accuracy'], 
            optimizer=optimizer)
        return model

    def train1(self):
        train_x, train_y = self.load_labeled_data()

        input_dim = train_x.shape[1]

        # if self.use_nn == False:
        #     print(train_x)
        #     model = svm.SVC(verbose = True)
        #     model.fit(train_x, train_y) 
        #     self.model = model
        #     joblib.dump(model, self.model_file_name) 

        #     print(model.score(test_x, test_y))
        #     return

        model = self.init_MLP_model(input_dim)

        # a stopping function should the validation loss stop improving
        # Add when validation set or ratio are supplied
        earlystop = EarlyStopping(monitor='val_loss', patience=1, verbose=0, mode='auto')

        # Tensorboard
        tb = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)

        num_epochs = 5
        batch_size = 32
        def fit_to_model():
            model.fit(train_x, 
                train_y,
                batch_size=batch_size, 
                nb_epoch=num_epochs, 
                shuffle=True,
                callbacks=[tb])
        fit_to_model()

        train_x = np.array([])
        train_y = np.array([])

        for file_id in range(0,20):
            print("File ID:", file_id)
            train_x_unlabeled = self.load_unlabeled_data_from_file(file_id)
            train_y_predicted = model.predict_proba(train_x_unlabeled, batch_size=batch_size)[:,1]
            # add: show the percentage accuracy in each prediction (you have the real labels)

            # Get the most confidently predicted features
            df = pd.DataFrame({"data":list(train_x_unlabeled), "pred":train_y_predicted}).sort_values(["pred"])
            qty = 100
            train_x_ = pd.concat([df.iloc[:qty], df.iloc[-qty:]])["data"].values
            train_x_ = np.array([v for v in train_x_])
            train_y_ = np.concatenate([np.zeros(qty), np.ones(qty)])
            train_y_ = np_utils.to_categorical(train_y_, 2)

            del model
            model = self.init_MLP_model(input_dim)
            train_x = np.concatenate([train_x, train_x_]) if train_x.size else train_x_
            train_y = np.concatenate([train_y, train_y_]) if train_y.size else train_y_
            fit_to_model()

            self.model = model
            self.validate(20)

        self.model = model
        self.model.save(self.model_file_name)

    def evaluate_model(self):
        test_x, test_y = self.load_labeled_data("test", 20)

        batch_size = 32
        
        scores = model.evaluate(test_x, test_y, verbose=0, batch_size=batch_size)
        print("Accuracy: %.2f%%" % (scores[1]*100))

        # Compute confusion matrix
        pred_y = self.model.predict_classes(test_x, batch_size=batch_size)
        cnf_matrix = confusion_matrix(np.array(test_y)[:,1], np.array(pred_y))
        np.set_printoptions(precision=2)

        # Plot non-normalized confusion matrix
        plt.figure()
        self.plot_confusion_matrix(cnf_matrix, classes=["non-HR", "HR"],
                              title='Confusion matrix, without normalization')

        # Plot normalized confusion matrix
        plt.figure()
        self.plot_confusion_matrix(cnf_matrix, classes=["non-HR", "HR"], normalize=True,
                              title='Normalized confusion matrix')

        plt.show()

    def train(self, train_anyway = False):
        if not train_anyway:
            print("A trained model already exists under %s, skipping." % self.model_file_name)
            return

        train_x, test_x, train_y, test_y = train_test_split(self.features, 
                                                            self.labels, 
                                                            test_size=0.2)

        if self.use_nn == False:
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
        num_epochs = 15
        batch_size = 32

        model = Sequential()
        model.add(Dense(128, input_dim=train_x.shape[1], activation="relu"))
        model.add(Dropout(0.2))
        #model.add(Dense(32, activation="relu"))
        #model.add(Dropout(0.2))
        model.add(Dense(2, activation="softmax"))

        optimizer = Adam(lr=learning_rate)
        model.compile(loss='categorical_crossentropy', 
            metrics=['accuracy'], 
            optimizer=optimizer)

        # a stopping function should the validation loss stop improving
        # Add when validation set or ratio are supplied
        earlystop = EarlyStopping(monitor='val_loss', patience=1, verbose=0, mode='auto')

        # Tensorboard
        tb = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)

        model.fit(train_x, 
            train_y,
            batch_size=batch_size, 
            nb_epoch=num_epochs, 
            shuffle=True,
            callbacks=[tb])
        
        scores = model.evaluate(test_x, test_y, verbose=0, batch_size=batch_size)
        print("Accuracy: %.2f%%" % (scores[1]*100))

        self.model = model
        self.model.save(self.model_file_name)


        # Compute confusion matrix
        pred_y = self.model.predict_classes(test_x, batch_size=batch_size)
        cnf_matrix = confusion_matrix(np.array(test_y)[:,1], np.array(pred_y))
        np.set_printoptions(precision=2)

        # Plot non-normalized confusion matrix
        plt.figure()
        self.plot_confusion_matrix(cnf_matrix, classes=["non-HR", "HR"],
                              title='Confusion matrix, without normalization')

        # Plot normalized confusion matrix
        plt.figure()
        self.plot_confusion_matrix(cnf_matrix, classes=["non-HR", "HR"], normalize=True,
                              title='Normalized confusion matrix')

        plt.show()


    def validate(self, file_id):

        if 0==1:

            assert self.nn == True, "Validate method currently only supported for NN implementation."

            #accuracies = np.array(self.model.predict_proba(self.features, batch_size=batch_size)[0])
            #df = pd.DataFrame(np.concatenate((accuracies, self.features), 1)) 

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
            
                psd = signal.extract_PSD_features(start, start+step)
                psd = self.scale_psd(psd)
                feat_vec = psd

                feat_vec = self.pca.transform(feat_vec)
                batch_size = 32
                acc = self.model.predict_proba(np.array([feat_vec]), batch_size=batch_size)
                acc = acc[0][0]

                ranges.append([acc, (start, start+step)])

                start += step 

            df = pd.DataFrame(ranges)

            df = df.sort_values(0)




        model = self.model
        batch_size = 32

        test_x, test_y = self.load_all_labeled_data("test", [20])
        batch_size = 32
        scores = model.evaluate(test_x, test_y, verbose=0, batch_size=batch_size)
        print("Accuracy: %.2f%%" % (scores[1]*100))

        test_x_unlabeled, signals = self.load_unlabeled_data_from_file(file_id, True)
        test_y_predicted = model.predict_proba(test_x_unlabeled, batch_size=batch_size)[:,1]
        # add: show the percentage accuracy in each prediction (you have the real labels)

        # Get the most confidently predicted features
        df = pd.DataFrame({"data":list(test_x_unlabeled), "pred":test_y_predicted}).sort_values(["pred"], ascending=False)

        num_figure_subplots = 30
        for k in range(1):
            plt.figure(k+1+self.plot_id)
            for i in range(num_figure_subplots):
                signal_id = df.iloc[k*num_figure_subplots + i].name
                signal = signals[signal_id]
                start = signal_id*256
                end = start+256

                signal_filtered = signal.bandpass_filter(0.8, 2.5)

                signal_scaled = preprocessing.scale(signal.content)
                signal_filtered_scaled = preprocessing.scale(signal_filtered)

                plt.subplot(num_figure_subplots/3,3,i+1)
                plt.plot(signal_scaled)
                plt.plot(signal_filtered_scaled, color='r')

                #peaks = scipy.signal.find_peaks_cwt(signal_scaled, np.arange(5,10))
                #plt.plot(peaks, signal_scaled[peaks], marker='o')
            plt.savefig('plots/plot%d.png' % self.plot_id)
            self.plot_id+=1

    def plot_confusion_matrix(self, cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        np.set_printoptions(precision=2)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm = np.around(cm, 2)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        print(cm)

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
