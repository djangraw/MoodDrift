#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CombineGbeConfirmResults.py

GbeConfirm was run in 5 splits so we wouldn't run out of memory on biowulf.
This script puts the results back together as if they were run all at once.

- Created 2/4/21 by DJ.
- Updated 2/18/21 by DJ - added doLate input for version excluding 1st rating.
- Updated 3/31/21 by DJ - adapted for shared code structure.
"""
# %%
# import packages
import pandas as pd
import numpy as np

dataDir = '../Data/OutFiles'
doLate = False # use '-late' suffix files that exluded first rating

# set up
nSplits = 5

# load results
dfParams_split = [0]*nSplits
predictions_split = [0]*nSplits
testLoss_split = [0]*nSplits
testLoss_noBetaT_split = [0]*nSplits
for iSplit in range(nSplits):
    if doLate:
        inFile= '%s/PyTorchParameters_GbeConfirm_split%d-late.csv'%(dataDir,iSplit)
    else:
        inFile= '%s/PyTorchParameters_GbeConfirm_split%d.csv'%(dataDir,iSplit)
    print('Loading %s...'%inFile)
    dfParams_split[iSplit] = pd.read_csv(inFile, index_col=0)


    if doLate:
        inFile= '%s/PyTorchPredictions_GbeConfirm_split%d-late.npy'%(dataDir,iSplit)
    else:
        inFile= '%/PyTorchPredictions_GbeConfirm_split%d.npy'%(dataDir,iSplit)
    print('Loading %s...'%inFile)
    predictions_split[iSplit] = np.load(inFile)

    # load tuning test results
    if doLate:
        inFile= '%s/PyTorchTestLoss_tune-GbeConfirm_split%d-late.npy'%(dataDir,iSplit)
    else:
        inFile= '%s/PyTorchTestLoss_tune-GbeConfirm_split%d.npy'%(dataDir,iSplit)
    print('Loading %s...'%inFile)
    testLoss_split[iSplit] = np.load(inFile)

    # load tuning test results w/ no betaT
    if doLate:
        inFile= '%s/PyTorchTestLoss_tune-GbeConfirm_split%d-late-NoBetaT.npy'%(dataDir,iSplit)
    else:
        inFile= '%s/PyTorchTestLoss_tune-GbeConfirm_split%d-NoBetaT.npy'%(dataDir,iSplit)
    print('Loading %s...'%inFile)
    testLoss_noBetaT_split[iSplit] = np.load(inFile)



# combine
print('Combining params...')
dfParams = pd.concat(dfParams_split,axis=0)
print('   %d subjects total.'%dfParams.shape[0])

print('Combining predictions...')
predictions = np.concatenate(predictions_split,axis=1)
print('   %d subjects total.'%predictions.shape[1])

print('Combining test loss...')
testLoss = np.concatenate(testLoss_split,axis=0)
print('   %d subjects total.'%testLoss.shape[0])

print('Combining noBetaT test loss...')
testLoss_noBetaT = np.concatenate(testLoss_noBetaT_split,axis=0)
print('   %d subjects total.'%testLoss_noBetaT.shape[0])

# save
if doLate:
    outFile = '%s/PyTorchParameters_GbeConfirm-late.csv'%(dataDir)
else:
    outFile = '%s/PyTorchParameters_GbeConfirm.csv'%(dataDir)
print('Saving combined params to %s...'%outFile)
dfParams.to_csv(outFile)
print('Done!')

if doLate:
    outFile = '%s/PyTorchPredictions_GbeConfirm-late.npy'%(dataDir)
else:
    outFile = '%s/PyTorchPredictions_GbeConfirm.npy'%(dataDir)
print('Saving combined predictions to %s...'%outFile)
np.save(outFile,predictions)
print('Done!')

# tuning test results
if doLate:
    outFile = '%s/PyTorchTestLoss_tune-GbeConfirm-late.npy'%(dataDir)
else:
    outFile = '%s/PyTorchTestLoss_tune-GbeConfirm.npy'%(dataDir)
print('Saving combined test loss to %s...'%outFile)
np.save(outFile,testLoss)
print('Done!')

# tuning test results w/ no betaT
if doLate:
    outFile = '%s/PyTorchTestLoss_tune-GbeConfirm-late-noBetaT.npy'%(dataDir)
else:
    outFile = '%s/PyTorchTestLoss_tune-GbeConfirm-noBetaT.npy'%(dataDir)
print('Saving combined test loss (no betaT) to %s...'%outFile)
np.save(outFile,testLoss_noBetaT)
print('Done!')
