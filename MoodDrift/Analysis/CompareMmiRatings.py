#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 15:17:04 2020

@author: jangrawdc

- Updated 10/29/20 by DJ - made axis labels have only first letter capitalized
- Updated 3/31/21 by DJ - adapted for shared code structure.
- Updated 8/11/22 by DJ - plot reference line at mean initial rating instead of 0.5
"""
# %%
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import MoodDrift.Analysis.PlotMmiData as pmd

def CompareMmiRatings(batchNames,procDataDir='../Data/OutFiles',batchLabels=[],iBlock='all',doInterpolation=False,makeNewFig=True):

    # Declare defaults
    if len(batchLabels)==0:
        batchLabels = batchNames

    # Set up figure
    if makeNewFig:
        plt.figure(511,figsize=(10,4),dpi=180);
        plt.clf();
    initialMood = np.ones(len(batchNames))*np.nan
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
        initialMood[iBatch] = dfRatingMean['rating'].values[0]

    # Annotate plot
    # plt.axhline(0.5,c='k',ls='--',zorder=1,label='neutral mood')
    plt.axhline(np.mean(initialMood),c='k',ls='--',zorder=1,label='mean initial mood')
    plt.legend()
    titleStr = 'Batch ' + ' vs. '.join(batchLabels) + ' comparison'
    plt.title(titleStr)
#    plt.tight_layout()
