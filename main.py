import os
import sys
import numpy as np
import pandas as pd
from classes.Signal import Signal
from classes.DataSource import DataSource
from classes.SignalClassifier import SignalClassifier

ds = DataSource()
# Load initial labeled training set T
labeled_ds = ds.load_or_process_labeled_dataset()
# Load entire (unlabeled) data set P
ds.load_or_process_entire_dataset()
# Remove T from P (i.e. P = P-T)
ds.remove_labeled_subset_from_dataset(labeled_ds)

# Initialize model
c = SignalClassifier()
input_dim = len(labeled_ds.feature_vec.iloc[0])
c.init_nn_model(input_dim=input_dim)

num_batches = 10
batch = 1
# Train on T
print("Batch %d/%d" % (batch, num_batches))
c.nn_batch_train(labeled_ds, num_epochs=10) 

while batch<num_batches:
    batch+=1
    # First, sort the dataset by model predictions
    ds.dataset = c.pred_and_sort(ds.dataset)
    
    qty = 100
    if ds.dataset.shape[0] < qty*2:
        break # we reached end of dataset

    # Extract the most confidently classified new features T from P
    most_confident_samples = pd.concat([ds.dataset.iloc[:qty], 
                                        ds.dataset.iloc[-qty:]])
    # Drop these from greater dataset (in memory only) to avoid 
    # using them in next iteration (P = P-T)
    samples_to_drop = list(ds.dataset.iloc[:qty].index.values) + \
                      list(ds.dataset.iloc[-qty:].index.values) 
    ds.dataset.drop(samples_to_drop, inplace=True)

    # Generate labels based on predictions
    labels = np.concatenate([np.ones(qty), np.zeros(qty)])
    most_confident_samples["label"] = list(labels)

    print("\r\nBatch %d/%d" % (batch, num_batches))
    c.nn_batch_train(most_confident_samples, num_epochs=4)

# Evaluate
test_ds = ds.load_or_process_labeled_dataset(from_file_id=20)
print("Positive test set size",test_ds[test_ds.label == 1].shape[0])
print("Negative test set size",test_ds[test_ds.label == 0].shape[0])
results = c.evaluate(test_ds)
results = {c.model.metrics_names[i]:v for i,v in enumerate(results)}
print(results)

# Display results
test_ds = c.pred_and_sort(test_ds)
ds.confusion(test_ds)
c.plot_losses()
# Uncomment for plots of most confidently predicted segments
ds.display_dataset(test_ds)
