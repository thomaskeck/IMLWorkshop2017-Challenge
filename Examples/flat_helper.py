from __future__ import print_function
import sys, os, math
import ROOT
import numpy as np
import pickle
import pandas as pd
import numpy as np
import root_numpy

from ROOT import TTree, TFile


# HELPER FUNCTIONS
###########################################################################################

class BreakLoop(Exception): pass

    
def GetJetShapes(rootFile, numSamples=-1, offset = 0, recompute = False):
    """Returns a data frame containing the Jet shapes
       rootFile can be a pattern (e.g. /mydir/*.root)
       The data frame is saved to a pickled file and reloaded from it (unless recompute is True)"""


    import pandas as pd
    import numpy as np

    pickleFileName = rootFile[1:rootFile.rfind('.')]+'.pkl'
    pickleFileName=pickleFileName.replace('/','_')
    pickleFileName=pickleFileName.replace('*','ALL')
    
    try:
        if(recompute):
            raise ValueError('Recompute')
        # if the file cannot be opened, this raises an exception which leads to the actual recomputation.
        with open(pickleFileName, 'rb') as fileP:
            df = pickle.load(fileP)
            print ("Loading from pickle file {0}".format(pickleFileName))
            return df
    except:
    
        # Get raw input data from delphes
        import glob
        chain = ROOT.TChain("treeJets")
        listOfFiles = glob.glob(rootFile)
        for fileIn in listOfFiles:
            chain.Add(fileIn)

        # The progress bar does not (yet) work on swan: commenting out.
        # try:
        #     # show progress
        #     from ipywidgets import FloatProgress
        #     from IPython.display import display
        #     progressBar = FloatProgress(min=0, max=100)
        #     display(progressBar)    
        #     show_progress = True
        # except:
        #     show_progress = False

      
        if numSamples < 0:
            numSamples = chain.GetEntries()
        data = pd.DataFrame(np.zeros((numSamples,4)), columns=['mass','ntowers','radial','dispersion'])

        
        # Loop over all events
        ijet = 0
        skipped = 0
        try:
            for event in chain:
                
                print('\r'+' Processing {0} [{1}]'.format(rootFile, float(ijet)/numSamples*100 ), end='')
                sys.stdout.flush()
                    
                # if show_progress:
                #     # Update progress bar
                #     progressBar.value = float(ijet)/numSamples*100    
                        
                # Do jet selection here
                if skipped < offset:
                    skipped += 1
                    continue

                # Fill data
                data.iloc[ijet] = CalculateJetShapes(event)

                ijet += 1
                if ijet >= numSamples:
                    raise BreakLoop
        except BreakLoop:
            pass


        print('\n')
        sys.stdout.flush()

        
        if ijet < numSamples:
            print('Only {:d} samples loaded (requested = {:d}). Not enough samples?'.format(ijet, numSamples))

        with open(pickleFileName, 'wb') as fileP:
            pickle.dump(data, fileP)
        
        return data


###########################################################################################
    
def GetJetShapesFast(rootFileDir, numSamples=-1, offset = 0, recompute = False):
    """Returns a data frame containing the Jet shapes
       rootFileDir should be a folder containing root files
       The jet shapes are saved to a root file and reloaded from it (unless recompute is True).
       This fast version uses a root macro to compute the shapes (iteration happens in C)"""

    
    #rootFileNameShapes = rootFileDir[1:rootFileDir.rfind('.')]+'_shapes.root'
    rootFileNameShapes = rootFileDir+'_shapes.root'
    rootFileNameShapes=rootFileNameShapes.replace('/','_')
    rootFileNameShapes=rootFileNameShapes.replace('*','ALL')

    #print(rootFileNameShapes)
    
    try:
        if(recompute):
            raise ValueError('Recompute')
        # if the file cannot be opened, this raises an exception which leads to the actual recomputation.
        data = GetShapesFromROOTFile(rootFileNameShapes)
        print ("Loading from root file {0}".format(rootFileNameShapes))
    except:

        # Compute shapes with external macro
        ROOT.gROOT.LoadMacro("CreateJetShapes.C")
        ROOT.CreateJetShapes(rootFileDir,rootFileNameShapes,numSamples)
        
        data = GetShapesFromROOTFile(rootFileNameShapes)    
        
    if len(data) < numSamples:
        print('Only {:d} samples loaded (requested = {:d}). Not enough samples?'.format(len(data), numSamples))

    return data



    

###########################################################################################
def CalculateJetShapes(entry):
    """Calculate jet shapes and add them to the jet instance"""
    leadingHadronPt    = -999.
    subleadingHadronPt = -999.
    jetDispersionSum = 0
    jetDispersionSquareSum = 0
    numConst = 0

    ShapeRadial = 0.


    ntracks = entry.ntracks
    for itrack in range(0,ntracks):
        if abs(entry.trackEta[itrack]) > 20.: #FIXME: Do we need this?
            continue

        # Get leading hadron pt
        if entry.trackPt[itrack] > leadingHadronPt:
            subleadingHadronPt = leadingHadronPt
            leadingHadronPt    = entry.trackPt[itrack]
        elif entry.trackPt[itrack] > subleadingHadronPt:
            subleadingHadronPt = entry.trackPt[itrack]

        deltaPhi = min(abs(entry.jetPhi-entry.trackPhi[itrack]), 2*math.pi- abs(entry.jetPhi-entry.trackPhi[itrack]))
        deltaEta = entry.jetEta-entry.trackEta[itrack]
        deltaR   = math.sqrt(deltaPhi*deltaPhi + deltaEta*deltaEta)

        # Calculate properties important for shape calculation
        jetDispersionSum += entry.trackPt[itrack]
        jetDispersionSquareSum += entry.trackPt[itrack]*entry.trackPt[itrack]
        ShapeRadial += entry.trackPt[itrack]/entry.jetPt * deltaR

        numConst += 1

    # Calculate the shapes
    if numConst > 1:
        ShapeLeSub = leadingHadronPt - subleadingHadronPt
    else:
        ShapeLeSub = 1.

    if jetDispersionSum:
        ShapeDispersion = math.sqrt(jetDispersionSquareSum)/jetDispersionSum
    else:    
        ShapeDispersion = 0.
        

    return [entry.jetMass, entry.ntowers, ShapeRadial, ShapeDispersion]


def GetShapesFromROOTFile(fileName):
    """Opens a root file containing a treeShapes tree and loads it into a numpy array"""

    f = TFile(fileName)
    t = f.Get("treeShapes")
    array = root_numpy.tree2array(t)
    return  pd.DataFrame(array)
    


