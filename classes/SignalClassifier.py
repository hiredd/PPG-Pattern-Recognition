import numpy as np
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, svm
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import tensorflow as tf
import os.path
import itertools
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.regularizers import l2, activity_l2
import keras.backend as K
import matplotlib.pyplot as plt
from classes.Signal import Signal

class SignalClassifier:

    model = None
    losses = []

    def init_nn_model(self, input_dim):
        # Network hyperparameters
        learning_rate = 10**(-2)
        dropout_rate = 0.2
        decay_rate = 0.05
        
        l1_neurons = 128
        l2_neurons = 32
        output_dim = 2

        model = Sequential()
        model.add(Dense(l1_neurons, input_dim=input_dim, activation="relu"))
        model.add(Dropout(dropout_rate))
        model.add(Dense(l2_neurons, activation="relu"))
        model.add(Dropout(dropout_rate))
        model.add(Dense(output_dim, activation="softmax"))

        optimizer = Adam(lr=learning_rate, decay=decay_rate)
        model.compile(loss='categorical_crossentropy', 
                      metrics=['accuracy'], 
                      optimizer=optimizer)
        self.model = model

    def k_mean_train(self, dataset):
        train_x = np.vstack(dataset["feature_vec"])
        train_y = np.array(dataset["label"].values, dtype=np.int8)

        train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=0.2)

        kmeans = KMeans(n_clusters=1).fit(train_x, train_y)
        print("kmeans", kmeans.score(train_x, train_y))

    def train(self, dataset, num_epochs=1, verbose=False):
        train_x = np.vstack(dataset["feature_vec"])
        train_y = self.one_hot_encode(dataset["label"].values)

        res = self.model.fit(train_x, 
                             train_y, 
                             batch_size=32, 
                             nb_epoch=num_epochs, 
                             shuffle=True, 
                             verbose=verbose)

        self.losses += res.history["loss"]

    def evaluate(self, dataset):
        test_x = np.vstack(dataset["feature_vec"])
        test_y = self.one_hot_encode(dataset["label"].values)
        return self.model.evaluate(test_x, test_y)

    def pred_and_sort(self, dataset):
        train_x = np.vstack(dataset["feature_vec"])
        y_predicted = self.model.predict_proba(train_x)[:,1]
        dataset["pred"] = list(y_predicted)
        # Get the most confidently predicted features
        data_sorted_by_preds = dataset.sort_values(["pred"], ascending=False)
        return data_sorted_by_preds

    def plot_losses(self):
        plt.figure()
        ax = plt.subplot(1,1,1)
        ax.plot(self.losses, color='black')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        plt.show()

    def one_hot_encode(self, y):
        y = np.array(y, dtype=np.int8)
        y = np_utils.to_categorical(y, 2)
        return y
