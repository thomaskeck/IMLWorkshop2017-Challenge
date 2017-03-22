#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Thomas Keck and Jochen Gemmler 2017

# We put our converted and flattened data into one big file for
# - training the boost network
# - training the inference network
# - test the inference network

import pandas

df_qm = pandas.read_pickle("quarks_modified_flat.pickle")
df_qs = pandas.read_pickle("quarks_standard_flat.pickle")
df_gm = pandas.read_pickle("gluons_modified_flat.pickle")
df_gs = pandas.read_pickle("gluons_standard_flat.pickle")

# Add truth column for the boost network
# here we want to train standard against modified events,
# to learn the difference between MC and "data"
df_qs['is_data'] = False
df_qm['is_data'] = True
df_gs['is_data'] = False
df_gm['is_data'] = True

df_boost = pandas.concat([df_qs, df_qm, df_gs, df_gm])
df_boost.to_pickle('boost_training_sample.pickle')
del df_boost
del df_qs['is_data']
del df_gs['is_data']
del df_qm['is_data']
del df_gm['is_data']

# Add truth column for the inference network
# quarks are considered signal
# gluons are considered background
df_qs['is_quark'] = True
df_gs['is_quark'] = False
df_inference = pandas.concat([df_qs, df_gs])
df_inference.to_pickle('inference_training_sample.pickle')
del df_inference

# Add truth column for the inference network
# quarks are considered signal
# gluons are considered background
df_qm['is_quark'] = True
df_gm['is_quark'] = False
df_inference_test = pandas.concat([df_qm, df_gm])
df_inference_test.to_pickle('inference_test_sample.pickle')
del df_inference_test
