import pandas
import numpy as np

variables = ['jetPt', 'jetEta', 'jetPhi', 'jetMass', 'ntracks', 'ntowers']
for v in ['trackPt', 'trackEta', 'trackPhi', 'trackCharge']:
    for i in range(1):
        variables += [v + '_' + str(i)]
for v in ['towerE', 'towerEem', 'towerEhad', 'towerEta', 'towerPhi']:
    for i in range(1):
        variables += [v + '_' + str(i)]

df = pandas.read_pickle('inference_training_sample.pickle')
data_signal = df[df.is_quark == 1][variables].values
data_background = df[df.is_quark != 1][variables].values
fisher_coefficients = np.dot(np.linalg.inv(np.cov(np.transpose(data_signal)) + np.cov(np.transpose(data_background))), np.mean(data_signal, 0) - np.mean(data_background, 0))

train_data = pandas.read_pickle('inference_training_sample.pickle')
x_train = train_data[variables].values
y_train = train_data['is_quark'].values
x_train = np.require(x_train, dtype=np.float32, requirements=['A', 'W', 'C', 'O'])
probability = np.dot(x_train, fisher_coefficients)
df_result_train = pandas.DataFrame({'y': y_train, 'p': probability})
df_result_train.to_pickle('result_train_fisher.pickle')

test_data = pandas.read_pickle('inference_test_sample.pickle')
x_test = test_data[variables].values
y_test = test_data['is_quark'].values
x_test = np.require(x_test, dtype=np.float32, requirements=['A', 'W', 'C', 'O'])
probability = np.dot(x_test, fisher_coefficients)
df_result_test = pandas.DataFrame({'y': y_test, 'p': probability})
df_result_test.to_pickle('result_test_fisher.pickle')
