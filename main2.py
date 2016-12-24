import os
import sys
from datetime import datetime 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from classes.Signal import Signal
from classes.DataSource import DataSource
from classes.HRClassifier2 import HRClassifier2

regenerate_dataset = False

ds = DataSource()
# Load initial labeled training set T
labeled_ds = ds.load_or_process_labeled_dataset()
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
ds.dataset.reset_index(drop=True, inplace=True)

# Multilayer perceptron
c = HRClassifier2()
input_dim = len(labeled_ds.feature_vec.iloc[0])
c.init_nn_model(input_dim=input_dim)

num_batches = 10
batch = 1
# Train on T
print("Batch %d/%d" % (batch, num_batches))
c.nn_batch_train(labeled_ds, num_epochs=10) # there is a problem with this - the predictions are not good enough for 1000 examples
# try just training and predicting (with fit and predict_proba) with the rest of the dataset - see what happens
#c.nn_train(labeled_ds)

while 1==0 and batch<num_batches:
    batch+=1
    # Use the model to classify new features T' from P, use the most 
    # confidently classified ones as the next training batch:
    # First, sort the dataset by model predictions
    ds.dataset = c.pred_and_sort(ds.dataset)

    #ds.display_dataset(ds.dataset)
    #quit()
    
    qty = 100
    if ds.dataset.shape[0] < qty*2:
        break

    # Extract the top most confidently classified samples
    most_confident_samples = pd.concat([ds.dataset.iloc[:qty], ds.dataset.iloc[-qty:]])
    samples_to_drop = list(ds.dataset.iloc[:qty].index.values) + list(ds.dataset.iloc[-qty:].index.values) 
    # Drop samples from greater dataset to avoid refitting them
    ds.dataset.drop(samples_to_drop, inplace=True)

    # Use feature vectors and generate labels as the next batch: T = T'
    labels = np.concatenate([np.ones(qty), np.zeros(qty)])
    most_confident_samples["label"] = list(labels)

    print("Batch %d/%d" % (batch, num_batches))
    c.nn_batch_train(most_confident_samples, num_epochs=4)
    #print("loss:", np.mean(losses), np.mean(accuracies))

# Evaluate
test_ds = ds.load_or_process_labeled_dataset(20)
print("Positive test set size",test_ds[test_ds.label == 1].shape[0])
print("Negative test set size",test_ds[test_ds.label == 0].shape[0])
metrics = {c.model.metrics_names[i]:v for i,v in enumerate(c.evaluate(test_ds))}
print(metrics)
test_ds.reset_index(drop=True, inplace=True)
test_ds = test_ds.reindex(np.random.permutation(test_ds.index))

# Display results
test_ds = c.pred_and_sort(test_ds)
ds.display_dataset(test_ds)
quit()

ds.dataset = c.pred_and_sort(ds.dataset)
ds.display_dataset(ds.dataset)