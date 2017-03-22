#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Thomas Keck and Jochen Gemmler 2017

import numpy as np
import tensorflow as tf
import pandas
import os

os.environ['CUDA_VISIBLE_DEVICES']='3'

# We use all available variables
# - Global variables of the jet
# - Track variables for each track sorted by decreasing transverse momentum
# - Tower variables for each tower sorted by decreasing energy
variables = ['jetPt', 'jetEta', 'jetPhi', 'jetMass', 'ntracks', 'ntowers']

for v in ['trackPt', 'trackEta', 'trackPhi', 'trackCharge']:
    for i in range(52):
        variables += [v + '_' + str(i)]

for v in ['towerE', 'towerEem', 'towerEhad', 'towerEta', 'towerPhi']:
    for i in range(67):
        variables += [v + '_' + str(i)]


def batch_generator(filename, target, batch_size):
    """
    Returns random batch from the datafile, and ensures
    that the numpy ndarrays have the correct format, to avoid
    any weird memory problems
    """
    df = pandas.read_pickle(filename)
    for i in range(52):
        df['trackEta_' + str(i)] -= df['jetEta']
        df['trackPhi_' + str(i)] -= df['jetPhi']
        df['trackPt_' + str(i)] /= df['jetPt']
    for i in range(67):
        df['towerEta_' + str(i)] -= df['jetEta']
        df['towerE_' + str(i)] /= df['jetPt']
        df['towerEem_' + str(i)] /= df['jetPt']
        df['towerEhad_' + str(i)] /= df['jetPt']

    while True:
        df = df.sample(frac=1).reset_index(drop=True)
        for pos in range(0, len(df)-batch_size-1, batch_size):
            x = df.loc[pos:pos + batch_size-1, variables].values
            y = df.loc[pos:pos + batch_size-1, target].values
            y = np.reshape(y, (len(y), 1))
            x = np.require(x, dtype=np.float32, requirements=['A', 'W', 'C', 'O'])
            y = np.require(y, dtype=np.float32, requirements=['A', 'W', 'C', 'O'])
            yield x, y


def get_model(x):
    """
    Returns our neural network model.
    4 Hidden Layers with sigmoid activation and dropout.
    This same network is used for the boosting and the inference network
    """
    def layer(x, shape, name, unit=tf.sigmoid):
        with tf.name_scope(name) as scope:
            weights = tf.Variable(tf.truncated_normal(shape, stddev=1.0 / np.sqrt(float(shape[0]))), name='weights')
            biases = tf.Variable(tf.constant(0.0, shape=[shape[1]]), name='biases')
            layer = unit(tf.matmul(x, weights) + biases)
        return layer

    dropout_const = .75
    hidden1 = layer(x, [len(variables), 400], 'hidden1')
    hidden1 = tf.nn.dropout(hidden1, dropout_const)
    hidden2 = layer(hidden1, [400, 400], 'hidden2')
    hidden2 = tf.nn.dropout(hidden2, dropout_const)
    hidden3 = layer(hidden2, [400, 400], 'hidden3')
    hidden3 = tf.nn.dropout(hidden3, dropout_const)
    hidden4 = layer(hidden3, [400, 400], 'hidden4')
    hidden4 = tf.nn.dropout(hidden4, dropout_const)
    activation = layer(hidden4, [400, 1], 'sigmoid', unit=tf.sigmoid)
    return activation


if __name__ == '__main__':

    n_iterations = int(1e6)
    use_boost = True
    epsilon = 1e-5
    
    x = tf.placeholder(tf.float32, [None, len(variables)], name='x')
    y = tf.placeholder(tf.float32, [None, 1], name='y')
    w = tf.placeholder(tf.float32, [None, 1], name='w')
    
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)

    # Boost model
    # This model is used to train standard against modified
    # The model learns the differences and outputs a probability to be "modified" (=data),
    # this probability is used to weight the inference training input,
    # this technique is known in literature to remove differences between data and MC
    boost_activation = get_model(x)
    loss_boost = -tf.reduce_sum(y * w * tf.log(boost_activation + epsilon) +
                                (1.0 - y) * w * tf.log(1 - boost_activation + epsilon)) / tf.reduce_sum(w)
    minimize_boost = optimizer.minimize(loss_boost)
    
    # Inference model
    # Trained to distinguish quarks from gluon jets
    inference_activation = get_model(x)
    loss = -tf.reduce_sum(y * w * tf.log(inference_activation + epsilon) +
                                (1.0 - y) * w * tf.log(1 - inference_activation + epsilon)) / tf.reduce_sum(w)
    minimize = optimizer.minimize(loss)
    
    # Initialise tensorflow
    init = tf.global_variables_initializer()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    session.run(init)
    
    # Train Boost Network
    if use_boost:
        saver = tf.train.Saver()
        
        batch = batch_generator('boost_training_sample.pickle', 'is_data', 200)

        for step in range(n_iterations // 10):

            batch_xs, batch_ys = next(batch)
            batch_ws = np.ones(len(batch_ys))
            # Correct signal to background fraction (see below)
            # We have 71% background (standard) and 29% signal (modified)
            batch_ws = np.where(batch_ys[:, 0] == 1, 0.5 * (0.71 / 0.29 + 1), 0.5 * (0.29 / 0.71 + 1)) * batch_ws
            batch_ws = np.reshape(batch_ws, (len(batch_ys), 1))
            batch_ws = np.require(batch_ws, dtype=np.float32, requirements=['A', 'W', 'C', 'O'])
            feed_dict = {x: batch_xs, y: batch_ys, w: batch_ws}
            _, loss_value = session.run([minimize_boost, loss_boost], feed_dict=feed_dict)

            if step % 500 == 0:
                print('Step %d: loss = %.2f' % (step, loss_value))

            if (step + 1) % 10000 == 0 or (step + 1) == n_iterations:
                print('Save model')
                saver.save(session, 'boost_model_transformed', global_step=step)

    
        del batch
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    tf.add_to_collection('p', inference_activation)

    saver = tf.train.Saver()

    # Train inference network
    batch = batch_generator('inference_training_sample.pickle', 'is_quark', 200)

    for step in range(n_iterations):
        batch_xs, batch_ys = next(batch)
        if use_boost:
            # We apply the boost network to calculate the weights,
            # the weight formula is w = p / (1-p)
            # see http://www-ekp.physik.uni-karlsruhe.de/~jwagner/www/publications/AdvancedReweighting_MVA_ACAT2011.pdf
            batch_ws = session.run(boost_activation, feed_dict={x: batch_xs})
            batch_ws = (batch_ws + epsilon) / (1 - batch_ws + epsilon)
            batch_ws = batch_ws[:, 0]
        else:
            batch_ws = np.ones(len(batch_ys))
        # We normalise the events, so that there is the same amount of signal-weight and background-weight
        # in the training, using the ratios we know from the standard datasets
        # 71% signal (quarks), 29% background (gluons)
        # The formula is:
        #   w_s = 1/2 * (1 + N_B / N_S)
        #   w_b = 1/2 * (1 + N_S / N_B)
        # The numbers are the opposite of the ones in the boosting network, this is by chance!
        batch_ws = np.where(batch_ys[:, 0] == 1, 0.5 * (0.29 / 0.71 + 1), 0.5 * (0.71 / 0.29 + 1)) * batch_ws
        batch_ws = np.reshape(batch_ws, (len(batch_ys), 1))
        batch_ws = np.require(batch_ws, dtype=np.float32, requirements=['A', 'W', 'C', 'O'])

        feed_dict = {x: batch_xs, y: batch_ys, w: batch_ws}
        _, loss_value = session.run([minimize, loss], feed_dict=feed_dict)

        if step % 500 == 0:
            print('Step %d: loss = %.2f' % (step, loss_value))

        if (step + 1) % 10000 == 0 or (step + 1) == n_iterations:
            print('Save model')
            saver.save(session, 'inference_model_final_transformed', global_step=step)
  
    del batch
