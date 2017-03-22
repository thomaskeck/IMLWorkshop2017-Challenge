#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Thomas Keck and Jochen Gemmler 2017

# Applies a tensorflow model

import pandas
import tensorflow as tf
import numpy as np
import sklearn
import sklearn.metrics

from tf_model import variables


def apply(X, session, x, p):
    result = session.run(p, feed_dict={x: X})
    prediction = result[:, 0]
    return prediction


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.ERROR)
    name = 'boost_model-99999'    
    with tf.Graph().as_default() as graph:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as session:
            saver = tf.train.import_meta_graph('./' + name + '.meta')
            tf.train.update_checkpoint_state('./', name)

            saver.restore(session, tf.train.latest_checkpoint('./'))

            x = graph.get_operation_by_name("x").outputs[0]
            weights = graph.get_operation_by_name("hidden1/weights").outputs[0] 
            biases = graph.get_operation_by_name("hidden1/biases").outputs[0] 
            pred = tf.sigmoid(tf.matmul(x, weights) + biases)
            weights = graph.get_operation_by_name("hidden2/weights").outputs[0] 
            biases = graph.get_operation_by_name("hidden2/biases").outputs[0] 
            pred = tf.sigmoid(tf.matmul(pred, weights) + biases)
            weights = graph.get_operation_by_name("hidden3/weights").outputs[0] 
            biases = graph.get_operation_by_name("hidden3/biases").outputs[0] 
            pred = tf.sigmoid(tf.matmul(pred, weights) + biases)
            weights = graph.get_operation_by_name("hidden4/weights").outputs[0] 
            biases = graph.get_operation_by_name("hidden4/biases").outputs[0] 
            pred = tf.sigmoid(tf.matmul(pred, weights) + biases)
            weights = graph.get_operation_by_name("sigmoid/weights").outputs[0] 
            biases = graph.get_operation_by_name("sigmoid/biases").outputs[0] 
            pred = tf.sigmoid(tf.matmul(pred, weights) + biases)

            # Run trained network on the training sample
            train_data = pandas.read_pickle('inference_training_sample.pickle')
            x_train = train_data[variables].values
            y_train = train_data['is_quark'].values
            x_train = np.require(x_train, dtype=np.float32, requirements=['A', 'W', 'C', 'O'])

            p = np.zeros(len(y_train))
            for i in range(0, len(y_train), 100000):
                e = i + 100000
                e = min(len(y_train), e)
                p[i:e] = apply(x_train[i:e], session, x, pred)
                if i == 0:
                    print(p[i:e])

            print("train", sklearn.metrics.roc_auc_score(y_train, p))
            del train_data

            # Run trained network on the test sample
            test_data = pandas.read_pickle('inference_test_sample.pickle')
            x_test = test_data[variables].values
            y_test = test_data['is_quark'].values
            x_test = np.require(x_test, dtype=np.float32, requirements=['A', 'W', 'C', 'O'])

            p = np.zeros(len(y_test))
            for i in range(0, len(y_test), 100000):
                e = i + 100000
                e = min(len(y_test), e)
                p[i:e] = apply(x_test[i:e], session, x, pred)
            
            # Save results and print out ROC value
            print("test", sklearn.metrics.roc_auc_score(y_test, p))
            del test_data
