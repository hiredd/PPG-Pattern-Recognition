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

class HRClassifier2:

    # Neural network hyperparameters
    learning_rate = 10**(-3)
    dropout_rate = 0.2
    num_epochs = 5
    batch_size = 32

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

        optimizer = Adam(lr=self.learning_rate)
        model.compile(loss='categorical_crossentropy', 
            metrics=['accuracy'], 
            optimizer=optimizer)
        self.model = model

    def nn_batch_train(self, train_x, train_y):
        shuffled_ids = numpy.random.permutation(len(train_x))
        train_x, train_y = train_x[shuffled_ids], train_y[shuffled_ids]
        losses, accuracies = self.model.train_on_batch(train_x, train_y)
        return losses, accuracies

    def sort_by_pred(self, x_unlabeled):
        y_predicted = model.predict_proba(x_unlabeled["feature_vec"].values, batch_size=self.batch_size)[:,1]
        x_unlabeled["pred"] = y_predicted

        # Get the most confidently predicted features
        data_sorted_by_preds = x_unlabeled.sort_values(["pred"])
        return data_sorted_by_preds








        def fit_to_model():
            model.fit(train_x, 
                train_y,
                batch_size=self.batch_size, 
                nb_epoch=num_epochs, 
                shuffle=True,
                callbacks=[tb])
        fit_to_model()

        train_x = np.array([])
        train_y = np.array([])

        for file_id in range(0,20):
            print("File ID:", file_id)
            train_x_unlabeled = self.load_unlabeled_data_from_file(file_id)
            train_y_predicted = model.predict_proba(train_x_unlabeled, batch_size=self.batch_size)[:,1]
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