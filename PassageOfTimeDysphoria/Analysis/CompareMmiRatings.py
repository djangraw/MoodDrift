#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 15:17:04 2020

@author: jangrawdc

- Updated 10/29/20 by DJ - made axis labels have only first letter capitalized
- Updated 3/31/21 by DJ - adapted for shared code structure.
"""
# %%
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import PlotMmiData as pmd

def CompareMmiRatings(batchNames,procDataDir='../Data/OutFiles',batchLabels=[],iBlock='all',doInterpolation=False):

    # Declare defaults
    if len(batchLabels)==0:
        batchLabels = batchNames

    # Set up figure
    plt.figure(511,figsize=(10,4),dpi=180);
    plt.clf();

    for iBatch,batchName in enumerate(batchNames):

        dfRating = pd.read_csv('%s/Mmi-%s_Ratings.csv'%(procDataDir,batchName))
        dfTrial = pd.read_csv('%s/Mmi-%s_Trial.csv'%(procDataDir,batchName))

        # Limit to block
        if iBlock!='all':
            dfRating = dfRating.loc[dfRating.iBlock==iBlock,:]
            dfTrial = dfTrial.loc[dfTrial.iBlock==iBlock,:]
        # Plot averages
        dfRatingMean = pmd.GetMeanRatings(dfRating,doInterpolation=doInterpolation)
        dfTrialMean = pmd.GetMeanTrials(dfTrial)

        # Plot results
        nSubj = len(np.unique(dfTrial.participant))
        ratingLabel = '%s (n=%d)'%(batchLabels[iBatch],nSubj)
        pmd.PlotMmiRatings(dfTrialMean,dfRatingMean,'line',autoYlim=True,
                           doBlockLines=False,ratingLabel=ratingLabel)

    # Annotate plot
    plt.axhline(0.5,c='k',ls='--',zorder=1,label='neutral mood')
    plt.legend()
    titleStr = 'Batch ' + ' vs. '.join(batchLabels) + ' comparison'
    plt.title(titleStr)
#    plt.tight_layout()
