import pandas
import numpy as np

def myargsort(x):
    return np.argsort(x)[::-1].astype(int)

for name in ['gluons_modified', 'gluons_standard', 'quarks_modified', 'quarks_standard']:
    print("Process file", name)
    df = pandas.read_pickle(name + '.pickle')

    # Flatten and zero-pad tracks
    maxtracks = 52 # df.ntracks.max()
    trackorder = df.trackPt.apply(myargsort).values

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
            
    # Save some memory
    for column in ['trackPt', 'trackEta', 'trackPhi', 'trackCharge']:
        del df[column]
    for column in ['towerE', 'towerEem', 'towerEhad', 'towerEta', 'towerPhi']:
        del df[column]
    
    df.to_pickle(name + '_flat.pickle')

