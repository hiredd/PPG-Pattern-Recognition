import os
import sys
from datetime import datetime 
import matplotlib.pyplot as plt
import numpy as np
import pandas
from scipy.signal import welch
from classes.Signal import Signal
from classes.DataSource import DataSource
from classes.HRClassifier2 import HRClassifier2

ds = DataSource()
# Load initial labeled training set T
features, labels = ds.load_labeled_data()
# Load entire (unlabeled) dataset P-T
ds.load_entire_dataset()

c = HRClassifier2()
# Multilayer perceptron
c.init_nn_model(input_dim=features.shape[1])

quit()

while True:
    # Train using T
    losses, accuracies = c.nn_batch_train(features, labels)

    # Use the model to classify new features T' from P-T, and use the most 
    # confidently classified ones as the next training batch:
    # First, sort the dataset by model predictions
    ds.dataset = c.sort_by_pred(ds.dataset)
    # Extract the top most confidently classified samples
    qty = 100
    most_confident_samples = pd.concat([df.iloc[:qty], df.iloc[-qty:]])
    samples_to_drop = list(df.iloc[:qty].index.values) + list(df.iloc[-qty:].index.values) 
    # Drop samples from greater dataset to avoid refitting them
    df.dataset.drop(samples_to_drop, inplace=True)

    # Use feature vectors and generate labels as the next batch: T = T'
    train_x = most_confident_samples["feature_vec"].values
    train_x = np.array([v for v in train_x])
    train_y = np.concatenate([np.zeros(qty), np.ones(qty)])
    train_y = np_utils.to_categorical(train_y, 2)

