import numpy as np
import pandas
import matplotlib.pyplot as plt
import sklearn
import sklearn.metrics
import sklearn.ensemble

def auc(df, label):
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(df['y'], df['p'])
    plt.plot(fpr, tpr, lw=4, label=label)
    return sklearn.metrics.roc_auc_score(df['y'], df['p'])

if __name__ == '__main__':
    df_train = pandas.read_pickle('result_train.pickle')
    df_test = pandas.read_pickle('result_test.pickle')
    df_train_boost = pandas.read_pickle('result_train_with_boost.pickle')
    df_test_boost = pandas.read_pickle('result_test_with_boost.pickle')
    df_train_fisher = pandas.read_pickle('result_train_fisher.pickle')
    df_test_fisher = pandas.read_pickle('result_test_fisher.pickle')

    print('Train', 'auc', auc(df_train, 'Train'))
    print('Test', 'auc', auc(df_test, 'Test'))
    print('Train with Boost', 'auc', auc(df_train_boost, 'Train with Boost'))
    print('Test with Boost', 'auc', auc(df_test_boost, 'Test with Boost'))
    print('Fisher Train', 'auc', auc(df_train_fisher, 'Fisher Train'))
    print('Fisher Test', 'auc', auc(df_test_fisher, 'Fisher Test'))
    plt.legend()
    plt.show()
