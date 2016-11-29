import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import tensorflow as tf
import pyspark.sql.types as ptypes
from pyspark import SparkConf, SparkContext, SQLContext
from pyspark.sql import HiveContext
from pyspark.sql.functions import col, collect_list, udf
from pyspark.sql import DataFrame
import pandas as pd
from classes.Signal import Signal
import os.path
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam, SGD
from keras.utils import np_utils

class HRClassifier:

    def load_data1(self):
        num_data_files = 21

        HR_ranges = pd.read_csv("data/HR_ranges.csv", header=None, names=["file_id", "start", "end"])
        non_HR_ranges = pd.read_csv("data/non_HR_ranges.csv", header=None, names=["file_id", "start", "end"])
        both_ranges = [non_HR_ranges, HR_ranges]

        for range_type, ranges in enumerate(both_ranges):

            if range_type == 0:
                print("Reading non-HR segments from files:")
            else:
                print("Reading HR segments from files:")

            for file_id in range(num_data_files):
                print(file_id)

                if ranges[ranges["file_id"] == file_id].size > 0:

                    data = pd.read_csv("data/data%d.csv" % file_id, header=0, index_col=0, dtype = {"device":np.int32,
                         "ppg":np.float32, 
                         "accx":np.float32, 
                         "accy":np.float32, 
                         "accz": np.float32})

                    for _,row in ranges[ranges["file_id"] == file_id].iterrows():
                        segment = data.iloc[row["start"]:row["end"]]
                        s = Signal(segment["ppg"].values, segment["timestamp"].values)

                        f, psd = s.log_PSD()

                        self.features.append(psd)
                        self.labels.append(range_type)

        self.features = np.array(self.features)
        self.features = preprocessing.scale(self.features)
        self.labels = np.array(self.labels)

    def __init__(self, only_psds = True):
        self.sample_window = 512
        self.model_file_name = 'model.ckpt'
        self.features = []
        self.labels = []
        self.only_psds = only_psds
        self.model = None

        range_file_schema = ptypes.StructType([
            ptypes.StructField("file_id", ptypes.IntegerType(), True),
            ptypes.StructField("start",   ptypes.IntegerType(), True),
            ptypes.StructField("end",     ptypes.IntegerType(), True)])

        self.data_file_schema = ptypes.StructType([
            ptypes.StructField("record_id", ptypes.IntegerType(), True),
            ptypes.StructField("timestamp", ptypes.StringType(), True),
            ptypes.StructField("device",    ptypes.IntegerType(), True),
            ptypes.StructField("ppg",       ptypes.DoubleType(), True),
            ptypes.StructField("accx",      ptypes.DoubleType(), True),
            ptypes.StructField("accy",      ptypes.DoubleType(), True),
            ptypes.StructField("accz",      ptypes.DoubleType(), True)])

        #self.HR_ranges = SQL_context.read.csv("data/HR_ranges.csv", header = False, schema=range_file_schema)
        #self.non_HR_ranges = SQL_context.read.csv("data/non_HR_ranges.csv", header = False, schema=range_file_schema)

    def load_data(self, SQL_context, sc, file_id):
        data = SQL_context.read.csv("data/data%d.csv" % file_id, header = True, schema=self.data_file_schema).select(["record_id", "ppg"])
        data.persist()
        sc.broadcast(data)

        r = self.HR_ranges.select(["start", "end"]).where(col("file_id") == file_id).rdd 
        r = r.map(lambda x: int(data.count()))


        #self.data.persist()

    def train1(self, validation_set_ratio = 0.2, validation_set = None, train_anyway = False):
        if not train_anyway:
            # Check for redundant training
            assert not os.path.exists(self.model_file_name), "A trained model already exists under %s." % self.model_file_name

        assert len(self.labels) > 0 and len(self.features) > 0, "Features and/or labels required for classifier."

        train_x, test_x, train_y, test_y = train_test_split(self.features, 
                                                            self.labels, 
                                                            test_size=0.2)
        
        train_y = np_utils.to_categorical(train_y, 2)
        test_y = np_utils.to_categorical(test_y, 2)

        if self.only_psds:
            # Hyperparameters
            learning_rate = 10**(-3)
            num_epochs = 20
            batch_size = 32
            n_inputs, n_input_features = np.shape(train_x)

            # NN layer parameters
            n_l1 = n_inputs//2
            n_l2 = n_inputs//2 
            n_outputs = 2

            model = Sequential()
            model.add(Dense(128, input_dim=train_x.shape[1], activation="relu"))
            model.add(Dropout(0.2))
            model.add(Dense(32, activation="relu"))
            model.add(Dropout(0.2))
            model.add(Dense(n_outputs, activation="softmax"))

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

    def train(self, validation_set_ratio = 0.2, validation_set = None, train_anyway = False):
        if not train_anyway:
            # Check for redundant training
            assert not os.path.exists(self.model_file_name), "A trained model already exists under %s." % self.model_file_name

        assert len(self.labels) > 0 and len(self.features) > 0, "Features and/or labels required for classifier."

        train_x, test_x, train_y, test_y = train_test_split(self.features, 
                                                            self.labels, 
                                                            test_size=0.2)
        
        if self.only_psds:
            #self.standardize_features()

            # Hyperparameters
            learning_rate = 0.001
            training_epochs = 15
            batch_size = 32
            n_inputs, n_input_features = np.shape(train_x)

            model, X, Y = self.MLP()

            # Loss and optimizer initialization
            cost_fn = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(model, Y))
            optimizer = tf.train.AdamOptimizer(
                learning_rate=learning_rate).minimize(cost_fn)

            init = tf.initialize_all_variables()
            saver = tf.train.Saver()

            with tf.Session() as sess:
                sess.run(init)
                for epoch in range(training_epochs):
                    ids = np.random.permutation(n_inputs)
                    avg_cost = 0.
                    n_batches = n_inputs//batch_size
                    for batch in range(n_batches):
                        batch_ids = ids[batch*batch_size:(batch+1)*batch_size]

                        _, c = sess.run([optimizer, cost_fn], feed_dict={
                            X: train_x[batch_ids],
                            Y: train_y[batch_ids]
                        })

                        avg_cost += c / n_batches

                    print(c)

            # Test model
            correct_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
            # Calculate accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            print("Accuracy:", accuracy.eval({X: test_x, Y: test_y}))

            # Save model
            saver.save(sess, self.model_file_name)
            self.model = sess

    def reset_data(self):
        self.features = np.array([])
        self.labels = np.array([])

    def predict(self, x):
        prediction = tf.argmax(self.Y, 1)
        return self.model.run(prediction, feed_dict={X: x})

    # Multilayer Perceptron NN for binary classification
    def MLP(self):

        # NN layer parameters
        n_inputs, n_input_features = np.shape(self.features)
        n_l1 = n_inputs//2
        n_l2 = n_inputs//2 
        n_outputs = 1

        layer_sizes = [n_input_features, n_l1, n_l2, n_outputs]

        X = tf.placeholder(tf.float32, [None, n_input_features])
        Y = tf.placeholder(tf.int32, [None,])

        inp = X
        for ls in range(len(layer_sizes)-1):
            W = tf.Variable(
                    tf.random_normal(
                        [layer_sizes[ls], layer_sizes[ls+1]]))
            b = tf.Variable(tf.zeros(layer_sizes[ls+1]))
            h = tf.add(tf.matmul(inp, W), b)
            if ls < len(layer_sizes)-2:
                h = tf.nn.relu(h)
            inp = h
        return h, X, Y














