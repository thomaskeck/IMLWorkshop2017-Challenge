#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Thomas Keck and Jochen Gemmler 2017

# We flatten our pickled dataframes,
# by adding 52*4 track columns (for each of the possible 52 tracks and 4 variables per track)
#    adding 67*5 tower columns (for each of the possible 67 towers and 5 variables per tower)
# The columns are sorted according to decreasing transverse momentum and energy for tracks and towers, respectively.
# If there are less tracks or towers the remaining columns are set to 0

import pandas
import numpy as np

def myargsort(x):
    """
    Sort the given ndarray, reverse the order and return the indices as integers
    We use this indices to sort our columns below
    """
    return np.argsort(x)[::-1].astype(int)

for name in ['gluons_modified', 'gluons_standard', 'quarks_modified', 'quarks_standard']:
    print("Process file", name)
    df = pandas.read_pickle(name + '.pickle')

    # Flatten and zero-pad tracks
    maxtracks = 52 # df.ntracks.max()
    trackorder = df.trackPt.apply(myargsort).values

    # TODO
    # Center tracks and towers to mean value of each jet
    # - Translate to largest pt axis
    # - Rotate to second largest pt axis
    # - Flip image

    for column in ['trackPt', 'trackEta', 'trackPhi', 'trackCharge']:
        print("Process column", column)
        zero_padded_array = np.zeros((len(trackorder), maxtracks))
        for i in range(len(trackorder)):
            zero_padded_array[i, :len(trackorder[i])] = df[column][i][trackorder[i]]
        for i in range(maxtracks):
            df[column + '_' + str(i)] = zero_padded_array[:, i]
    
    # Flatten and zero-pad towers
    maxtowers = 67 # df.ntowers.max()
    towerorder = df.towerE.apply(myargsort).values
    for column in ['towerE', 'towerEem', 'towerEhad', 'towerEta', 'towerPhi']:
        print("Process", column)
        zero_padded_array = np.zeros((len(towerorder), maxtowers))
        for i in range(len(towerorder)):
            zero_padded_array[i, :len(towerorder[i])] = df[column][i][towerorder[i]]
        for i in range(maxtowers):
            df[column + '_' + str(i)] = zero_padded_array[:, i]
            
    # Save some memory, by removing the original columns containing the arrays converted from root
    for column in ['trackPt', 'trackEta', 'trackPhi', 'trackCharge']:
        del df[column]
    for column in ['towerE', 'towerEem', 'towerEhad', 'towerEta', 'towerPhi']:
        del df[column]
   
    # Save the file with the postfix _flat, for flattened
    df.to_pickle(name + '_flat.pickle')

