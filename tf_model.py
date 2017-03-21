#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Thomas Keck 2017

import numpy as np
import tensorflow as tf
import pandas


variables = ['jetPt', 'jetEta', 'jetPhi', 'jetMass', 'ntracks', 'ntowers']
for v in ['trackPt', 'trackEta', 'trackPhi', 'trackCharge']:
    for i in range(52):
        variables += [v + '_' + str(i)]
for v in ['towerE', 'towerEem', 'towerEhad', 'towerEta', 'towerPhi']:
    for i in range(67):
        variables += [v + '_' + str(i)]


def batch_generator(filename, target, batch_size):
    df = pandas.read_pickle(filename)
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
    def layer(x, shape, name, unit=tf.sigmoid):
        with tf.name_scope(name) as scope:
            weights = tf.Variable(tf.truncated_normal(shape, stddev=1.0 / np.sqrt(float(shape[0]))), name='weights')
            biases = tf.Variable(tf.constant(0.0, shape=[shape[1]]), name='biases')
            layer = unit(tf.matmul(x, weights) + biases)
        return layer

    # Boost network
    hidden1 = layer(x, [len(variables), 200], 'hidden1')
    hidden2 = layer(hidden1, [200, 200], 'hidden2')
    hidden3 = layer(hidden2, [200, 200], 'hidden3')
    hidden4 = layer(hidden3, [200, 200], 'hidden4')
    activation = layer(hidden4, [200, 1], 'sigmoid', unit=tf.sigmoid)
    return activation


if __name__ == '__main__':

    n_iterations = int(1e5)
    use_boost = True
    
    x = tf.placeholder(tf.float32, [None, len(variables)], name='x')
    y = tf.placeholder(tf.float32, [None, 1], name='y')
    w = tf.placeholder(tf.float32, [None, 1], name='w')
    

    boost_activation = get_model(x)
    
    epsilon = 1e-5
    loss_boost = -tf.reduce_mean(y * tf.log(boost_activation + epsilon) +
                                (1.0 - y) * tf.log(1 - boost_activation + epsilon))
  
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    minimize_boost = optimizer.minimize(loss_boost)
    
    inference_activation = get_model(x)
    
    loss = -tf.reduce_sum(y * w * tf.log(inference_activation + epsilon) +
                                (1.0 - y) * w * tf.log(1 - inference_activation + epsilon)) / tf.reduce_sum(w)
  
    minimize = optimizer.minimize(loss)
    
    init = tf.global_variables_initializer()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    session.run(init)
    
    if use_boost:
        saver = tf.train.Saver()
        
        batch = batch_generator('boost_training_sample.pickle', 'is_data', 100)

        for step in range(n_iterations):

            batch_xs, batch_ys = next(batch)
            feed_dict = {x: batch_xs, y: batch_ys}
            _, loss_value = session.run([minimize_boost, loss_boost], feed_dict=feed_dict)

            if step % 500 == 0:
                print('Step %d: loss = %.2f' % (step, loss_value))

            if (step + 1) % 10000 == 0 or (step + 1) == n_iterations:
                print('Save model')
                saver.save(session, 'boost_model', global_step=step)

    
    saver = tf.train.Saver()

    batch = batch_generator('inference_training_sample.pickle', 'is_quark', 100)

    for step in range(n_iterations):
        batch_xs, batch_ys = next(batch)
        if use_boost:
            batch_ws = session.run(boost_activation, feed_dict={x: batch_xs})
            batch_ws = (batch_ws + epsilon) / (1 - batch_ws + epsilon)
        else:
            batch_ws = np.ones(len(batch_ys))
        batch_ws = np.reshape(batch_ws, (len(batch_ys), 1))
        batch_ws = np.require(batch_ws, dtype=np.float32, requirements=['A', 'W', 'C', 'O'])

        feed_dict = {x: batch_xs, y: batch_ys, w: batch_ws}
        _, loss_value = session.run([minimize, loss], feed_dict=feed_dict)

        if step % 500 == 0:
            print('Step %d: loss = %.2f' % (step, loss_value))

        if (step + 1) % 10000 == 0 or (step + 1) == n_iterations:
            print('Save model')
            saver.save(session, 'inference_model', global_step=step)
   
    train_data = pandas.read_pickle('inference_training_sample.pickle')
    x_train = train_data[variables].values
    y_train = train_data['is_quark'].values
    x_train = np.require(x_train, dtype=np.float32, requirements=['A', 'W', 'C', 'O'])
    probability = session.run(inference_activation, feed_dict={x: x_train})
    df_result_train = pandas.DataFrame({'y': y_train, 'p': probability[:, 0]})
    if use_boost:
        df_result_train.to_pickle('result_train_with_boost.pickle')
    else:
        df_result_train.to_pickle('result_train.pickle')

    test_data = pandas.read_pickle('inference_test_sample.pickle')
    x_test = test_data[variables].values
    y_test = test_data['is_quark'].values
    x_test = np.require(x_test, dtype=np.float32, requirements=['A', 'W', 'C', 'O'])
    probability = session.run(inference_activation, feed_dict={x: x_test})
    df_result_test = pandas.DataFrame({'y': y_test, 'p': probability[:, 0]})
    if use_boost:
        df_result_test.to_pickle('result_test_with_boost.pickle')
    else:
        df_result_test.to_pickle('result_test.pickle')

