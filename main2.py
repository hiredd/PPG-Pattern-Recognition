import os
import sys
from datetime import datetime 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from classes.Signal import Signal
from classes.DataSource import DataSource
from classes.HRClassifier2 import HRClassifier2

ds = DataSource()
# Load initial labeled training set T
labeled_ds = ds.load_labeled_dataset()
# Load entire (unlabeled) data set P
ds.load_or_process_entire_dataset()
# Remove T from P (i.e. P = P-T)
for i in range(labeled_ds.shape[0]):
    file_id = labeled_ds.iloc[i]["file_id"]
    start = labeled_ds.iloc[i]["start"] 
    end = labeled_ds.iloc[i]["end"]
    ds.dataset = ds.dataset[~((ds.dataset.file_id == file_id) & 
                            (ds.dataset.start == start) & 
                            (ds.dataset.end == end))]
c = HRClassifier2()
# Multilayer perceptron
input_dim = len(labeled_ds.feature_vec.iloc[0])
c.init_nn_model(input_dim=input_dim)

num_batches = 20
batch = 1
# Train using T
print("Batch %d/%d" % (batch, num_batches))
losses, accuracies = c.nn_batch_train(labeled_ds, num_epochs=5)

while batch<num_batches:
    batch+=1
    # Use the model to classify new features T' from P, use the most 
    # confidently classified ones as the next training batch:
    # First, sort the dataset by model predictions
    ds.dataset = c.pred_and_sort(ds.dataset)
    
    qty = 100
    if ds.dataset.shape[0] < qty*2:
        break

    # Extract the top most confidently classified samples
    most_confident_samples = pd.concat([ds.dataset.iloc[:qty], ds.dataset.iloc[-qty:]])
    samples_to_drop = list(ds.dataset.iloc[:qty].index.values) + list(ds.dataset.iloc[-qty:].index.values) 
    # Drop samples from greater dataset to avoid refitting them
    ds.dataset.drop(samples_to_drop, inplace=True)

    # Use feature vectors and generate labels as the next batch: T = T'
    labels = np.concatenate([np.zeros(qty), np.ones(qty)])
    most_confident_samples["label"] = list(labels)

    print("Batch %d/%d" % (batch, num_batches))
    losses, accuracies = c.nn_batch_train(most_confident_samples, num_epochs=2)
    print("loss:", np.mean(losses), np.mean(accuracies))

# Evaluate
test_ds = ds.load_labeled_dataset(20)
print("Test dataset size",test_ds.shape)
d = {c.model.metrics_names[i]:v for i,v in enumerate(c.evaluate(test_ds))}
print(d)

# Display results
test_ds = c.pred_and_sort(test_ds)
ds.display_dataset(test_ds)
