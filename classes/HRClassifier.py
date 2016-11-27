import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import tensorflow as tf

class HRClassifier:

    def __init__(self, only_psds = False):
        self.sample_window = 512
        self.model_file_name = 'model.ckpt'
        self.features = np.array([])
        self.labels = np.array([])
        self.only_psds = only_psds
        self.model = None

    def i_freq(self, lf, frequency):
        i = 0
        while lf[i] < frequency:
            i += 1
        if i == 0:
            return 0
        return i-1

    def standardize_features(self):
        self.features = preprocessing.scaler.fit(self.features)\
                                            .transform(self.features)


    def extract_features(self, signal, is_valid_HR = False, start = None, end = None):
        assert signal!=None, "No signal to extract features from."
        assert self.sample_window!=0, "Sample window must be specified and be > 0."
        if end:
            assert end <= len(signal.content), "Segment end index must be smaller than signal length."

        if start == None:
            start = 0
        if end == None:
            end = len(signal.content) 

        feature_set = []

        while start + self.sample_window < end:

            f, psds = signal.log_PSD(start, start + self.sample_window)

            if self.only_psds:
                # Feature vector is simply the low-pass filtered PSDs
                feature_set.append(psds[self.i_freq(f,1):])
            else:
                feature_vector = []

                # F1: Mean amplitude of HF components (>20Hz)
                feature_vector.append(np.mean(psds[self.i_freq(f,20):]))

                # F2: LF/HF component magnitude ratio (1-2Hz)/(4-5Hz)
                feature_vector.append(
                    np.sum(psds[self.i_freq(f,1):self.i_freq(f,2)+1]) /\
                    np.sum(psds[self.i_freq(f,4):self.i_freq(f,5)+1]))

                # F3: Peak variance
                BPM, peak_variance = signal.getBPM(start, start + self.sample_window)
                feature_vector += [peak_variance, BPM]

                # Add feature vector's range indices
                feature_vector += [start, start + self.sample_window]

                feature_set.append(np.array(feature_vector))

            start += self.sample_window

        np.append(self.features, np.array(feature_set))
        if is_valid_HR:
            np.append(self.labels, np.ones(len(feature_set)))
        else:
            np.append(self.labels, np.zeros(len(feature_set)))

    def train(self, validation_set_ratio = 0.2, validation_set = None, train_anyway = False):
        if not train_anyway:
            # Check for redundant training
            assert not Path(self.model_file_name).is_file(), "A trained model already exists under %s." % self.model_file_name

        assert len(self.labels) > 0 and len(self.features) > 0, "Features and/or labels required for classifier."

        train_x, test_x, train_y, test_y = train_test_split(self.features, 
                                                            self.labels, 
                                                            test_size=0.2)
        if self.only_psds:
            self.standardize_features()

            # Hyperparameters
            learning_rate = 1e-3
            training_epochs = 15
            batch_size = 64
            n_inputs, n_input_features = np.shape(self.features)

            model = self.MLP()

            # Loss and optimizer initialization
            cost_fn = fn.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(model, Y))
            optimizer = tf.train.AdamOptimizer(
                learning_rate=learning_rate).minimize(cost_fn)

            init = tf.initialize_all_variables()
            saver = tf.train.Saver()

            sess = tf.Session()
            sess.run(init)
            for epoch in training_epochs:
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

                print("Loss %f" % avg_cost)

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

        X = tf.placeholder(tf.float32, [None, n_input])
        self.X = X
        Y = tf.placeholder(tf.float32, [None, 1])
        self.Y = Y

        inp = X
        for ls in range(len(layer_sizes)-1):
            W = tf.Variable(
                    tf.random_normal(
                        [layer_sizes[ls], layer_sizes[ls+1]]))
            b = tf.Variable(tf.zeros(layer_sizes[ls+1]))
            h = tf.nn.relu(tf.add(tf.matmul(inp, W), b))
            inp = h
        return h














