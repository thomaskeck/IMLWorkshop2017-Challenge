#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Thomas Keck and Jochen Gemmler 2017

import pandas
import numpy as np
import sklearn
import sklearn.discriminant_analysis

# We do not use all variables in fisher,
# because this would lead to a singular covariance matrix
# and the method fails.
variables = ['jetPt', 'jetEta', 'jetPhi', 'jetMass', 'ntracks', 'ntowers']
for v in ['trackPt', 'trackEta', 'trackPhi', 'trackCharge']:
    for i in range(52):
        variables += [v + '_' + str(i)]
for v in ['towerE', 'towerEem', 'towerEhad', 'towerEta', 'towerPhi']:
    for i in range(67):
        variables += [v + '_' + str(i)]

train_data = pandas.read_pickle('inference_training_sample.pickle')
x_train = train_data[variables].values
y_train = train_data['is_quark'].values

fisher = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
fisher.fit(x_train, y_train)

x_train = np.require(x_train, dtype=np.float32, requirements=['A', 'W', 'C', 'O'])
probability = fisher.predict_proba(x_train)[:, 1]
df_result_train = pandas.DataFrame({'y': y_train, 'p': probability})
df_result_train.to_pickle('result_train_fisher.pickle')

test_data = pandas.read_pickle('inference_test_sample.pickle')
x_test = test_data[variables].values
y_test = test_data['is_quark'].values
x_test = np.require(x_test, dtype=np.float32, requirements=['A', 'W', 'C', 'O'])
probability = fisher.predict_proba(x_test)[:, 1]
df_result_test = pandas.DataFrame({'y': y_test, 'p': probability})
df_result_test.to_pickle('result_test_fisher.pickle')
