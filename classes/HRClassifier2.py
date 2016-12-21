import numpy as np
import pandas as pd
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
import keras.backend as K
import matplotlib.pyplot as plt

class HRClassifier2:

    # Neural network hyperparameters
    learning_rate = 10**(-2)
    dropout_rate = 0.2
    decay_rate = 0.05

    def __init__(self, use_nn=True):
        self.use_nn = use_nn
        self.plot_id = 0

        if use_nn:
            self.model_file_name = 'models/trained_model.h5'
            if os.path.exists(self.model_file_name):
                self.model = load_model(self.model_file_name)
        else:
            self.model_file_name = 'models/trained_model.pkl'
            if os.path.exists(self.model_file_name):
                self.model = joblib.load(self.model_file_name)

    def init_nn_model(self, input_dim):
        model = Sequential()
        model.add(Dense(128, input_dim=input_dim, activation="relu"))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(2, activation="softmax"))

        optimizer = Adam(lr=self.learning_rate, decay=self.decay_rate)
        model.compile(loss='categorical_crossentropy', 
            metrics=['accuracy'], 
            optimizer=optimizer)
        self.model = model

    def nn_batch_train(self, dataset, num_epochs=1, verbose=False):
        train_x = np.array([v for v in dataset["feature_vec"].values])
        train_y = np.array(dataset["label"].values, dtype=np.int8)
        train_y = np_utils.to_categorical(train_y, 2)

        for epoch in range(num_epochs):
            shuffled_ids = np.random.permutation(len(train_x))
            train_x, train_y = train_x[shuffled_ids], train_y[shuffled_ids]
            losses, accuracies = self.model.train_on_batch(train_x, train_y)

        if verbose:
            opt = self.model.optimizer
            exact_lr = K.get_value(opt.lr) * (1.0 / (1.0 + K.get_value(opt.decay) * K.get_value(opt.iterations)))
            print("Learning rate:",exact_lr)

        return losses, accuracies

    def evaluate(self, dataset):
        test_x = np.array([v for v in dataset["feature_vec"].values])
        test_y = np.array(dataset["label"].values, dtype=np.int8)
        test_y = np_utils.to_categorical(test_y, 2)
        return self.model.test_on_batch(test_x, test_y)

    def pred_and_sort(self, dataset):
        train_x = np.array([v for v in dataset["feature_vec"].values])

        y_predicted = self.model.predict_proba(train_x)[:,1]
        dataset["pred"] = list(y_predicted)

        # Get the most confidently predicted features
        data_sorted_by_preds = dataset.sort_values(["pred"])
        return data_sorted_by_preds


