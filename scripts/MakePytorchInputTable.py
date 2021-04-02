#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MakePytorchInputTable.py

Prepare data for processing with pytorch scripts.

- Created 10/29/20 by DJ.
- Updated 2/4/21 by DJ - split GbeConfirm dataset to avoid memory error
- Updated 3/31/21 by DJ - adapted for shared code structure, added call to
   RemoveFirstRatingFromPytorchData.
- Updated 4/2/21 by DJ - fixed typos.
"""

# Import packages
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from PassageOfTimeDysphoria.Analysis import RemoveFirstRatingFromPytorchData

# Declare folders
dataDir = '../Data/OutFiles'
batchNames = ['GbeExplore','GbeConfirm'] # batches to remove first rating from

# Load
for batchName in batchNames:
    dfRating = pd.read_csv('%s/Mmi-%s_Ratings.csv'%(dataDir,batchName),index_col=0)
    dfTrial = pd.read_csv('%s/Mmi-%s_Trial.csv'%(dataDir,batchName),index_col=0)

    # Add first rating to each participant's trials
    participants = np.unique(dfTrial.participant)
    for i,participant in enumerate(participants):
        dfTrial.loc[(dfTrial.iTrial==0) & (dfTrial.participant==participant),'rating'] = \
            dfRating.loc[(dfRating.iTrial==-1) & (dfRating.participant==participant),'rating']

    # Crop & Rename columns
    dfData = dfTrial.loc[:,['participant','iTrial','time','outcomeAmount','rating']]
    dfData.columns = ['participant','trial_no','time','outcomeAmount','happySlider.response']
    dfData = dfData.reset_index(drop=True);

    # Save results
    outFile = '%s/Mmi-%s_TrialForMdls.csv'%(dataDir,batchName)
    print('Saving as %s...'%outFile)
    dfData.to_csv(outFile)
    print('Done!')

    # print parameters
    nSubj = len(participants)
    nTrials = dfData.shape[0]/nSubj
    nRatings = np.sum(~np.isnan(dfData.loc[(dfData.participant==participants[1]),'happySlider.response']))
    nTraining = np.ceil(nRatings*3.0/4.0)
    print('=== TUNE COMMAND ===')
    print('python Tune_GBE_Pytorch.py -df %s -ns %d -nt %d -ntr %d -of outputFiles -os _tune-%s'%(outFile,nSubj,nTrials,nTraining,batchName))
    print('=== RUN COMMAND ===')
    print('python Run_GBE_Pytorch.py -df %s -ns %d -nt %d -of outputFiles -os _%s'%(outFile,nSubj,nTrials,batchName))

    # %% If it's big, split it
    nTrials = 30
    nSubj= dfData.shape[0]/nTrials
    print('%d subjects'%nSubj)
    nSplits = int(np.ceil(nSubj/5000.0))
    if nSplits>1:
        print('Splitting into %d data files...'%nSplits)
        for iSplit in range(nSplits):
            if iSplit<(nSplits-1):
                dfDataSplit = dfData.iloc[iSplit*5000*nTrials:(iSplit+1)*5000*nTrials,:]
            else:
                dfDataSplit = dfData.iloc[iSplit*5000*nTrials:,:]
            print('%d subjects in split %d'%(dfDataSplit.shape[0]/nTrials,iSplit))
            outFile = '%s/Mmi-%s_TrialForMdls_split%d.csv'%(dataDir,batchName,iSplit)
            print('Saving as %s...'%outFile)
            dfDataSplit.to_csv(outFile)
            print('Done!')



    # Remove first ratings to create '-late' versions
    if batchName=='GbeConfirm':
        for split in np.arange(4):
            RemoveFirstRatingFromPytorchData(batchName,split,dataDir)
    else:
        RemoveFirstRatingFromPytorchData(batchName,split=None,dataDir=dataDir)
