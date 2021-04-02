#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Remove first rating to create -late versions of data files for pytorch analysis.

- Created 12/18/20 by DJ.
- Updated 3/31/21 by DJ - made into function, adapted for shared code structure.
"""

# %%
import pandas as pd
import numpy as np

# Define main function
def RemoveFirstRatingFromPytorchData(batchName,split=None,dataDir='../Data/OutFiles'):

    if split is not None:
        inFile = '%s/Mmi-%s_TrialForMdls_split%d.csv'%(dataDir,batchName,split)
        outFile = '%s/Mmi-%s_TrialForMdls_split%d-late.csv'%(dataDir,batchName,split)
    else:
        inFile = '%s/Mmi-%s_TrialForMdls.csv'%(dataDir,batchName)
        outFile = '%s/Mmi-%s_TrialForMdls-late.csv'%(dataDir,batchName)
    print('Loading %s...'%inFile)
    dfData = pd.read_csv(inFile,index_col=0)
    participants = np.unique(dfData.participant)
    print('Removing first rating for each subject...')
    for participant in participants:
        dfData.loc[(dfData.participant==participant) & (dfData.trial_no==0),'happySlider.response'] = np.nan
        dfData.loc[(dfData.participant==participant) & (dfData.trial_no==0),'happySlider.rt'] = np.nan

    # Save result
    print('Saving to %s...'%outFile)
    dfData.to_csv(outFile)
    print('Done!')
