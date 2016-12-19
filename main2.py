import os
import sys
from datetime import datetime 
import matplotlib.pyplot as plt
import numpy as np
import pandas
from scipy.signal import welch
from classes.Signal import Signal
from classes.DataSource import DataSource
from classes.HRClassifier import HRClassifier2

ds = DataSource()
features, labels = ds.load_labeled_data()
entire_features = ds.load_entire_dataset()

c = HRClassifier2()
c.init_nn_model(input_dim=features.shape[1])
c.nn_batch_train(features, labels)
