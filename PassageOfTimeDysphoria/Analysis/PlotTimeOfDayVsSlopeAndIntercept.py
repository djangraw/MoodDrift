#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PlotTimeOfDayVsSlopeAndIntercept.py
Created on Thu Oct  1 13:36:08 2020

@author: jangrawdc

- Updated 10/29/20 by DJ - made axis labels have only first letter capitalized
- Updated 3/31/21 by DJ - adapted for shared code structure.
- Updated 4/15/21 by DJ - turned into function
- Updated 9/23/22 by DJ - ignore "follow-up" batches on MW and Boredom
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
import os.path

def PlotTimeOfDayVsSlopeAndIntercept(bigBatch = 'AllOpeningRestAndRandom'):

    # %% Load results from whole LME cohort
    # Declare directories
    dataCheckDir = '../Data/DataChecks' # where datacheck .csv files can be found
    procDataDir = '../Data/OutFiles' # where processed data is found
    outFigDir = '../Figures' # where output figures should go

    print('Loading data files...')
    # Load Pymer coefficients
    dfCoeffs = pd.read_csv('%s/Mmi-%s_pymerCoeffs-full.csv'%(procDataDir,bigBatch))

    # Get names of all included batches, as in RunPymerOnCovidData_Aug2020
    dfBatches = pd.read_csv('%s/Mmi-Batches.csv'%procDataDir)
    dfBatch = dfBatches[['batchName','ratingsFile','surveyFile','trialFile','lifeHappyFile','block0_type','nPreviousRuns']]
    dfBatch = dfBatch.loc[(dfBatches.block0_type=='rest') | (dfBatches.block0_type=='random'),:]
    dfBatch = dfBatch.loc[(dfBatch.nPreviousRuns==0),:]
    dfBatch = dfBatch.set_index('batchName')
    dfBatch = dfBatch.drop(['Stability01-random','Stability02-random','RecoveryNimh-run3'],axis=0,errors='ignore') # ignore rejected batches if they exist
    # exclude follow-up batches
    dfBatch = dfBatch.drop(['BoredomAfterOnly','BoredomBeforeAndAfter','MwAfterOnly','MwBeforeAndAfter'],axis=0,errors='ignore') # ignore rejected batches if they exist


    # %% Add time of day to each line, batch by batch

    for batchName,row in dfBatch.iterrows():
        # load data check file
        if batchName.startswith('RecoveryNimh'): # all runs have the same dataCheck file
            dataCheckFile = '%s/RecoveryNimh_dataCheck.csv'%(dataCheckDir)
        else:
            dataCheckFile = '%s/%s_dataCheck.csv'%(dataCheckDir,batchName)
        print('Loading times from file %s...'%dataCheckFile)
        if os.path.exists(dataCheckFile):
            dfDataCheck = pd.read_csv(dataCheckFile);
        else:
            print('Data check file for %s is missing. Skipping...'%batchName)
            continue

        # get particiapants list
        if 'Nimh-run1' in batchName:
            print('Making participant numbers in batch %s negative to avoid overlap with MTurk subjects.'%batchName)
            dfDataCheck.participant = -dfDataCheck.participant # make negative to avoid overlap with MTurk participant numbers
            dfDataCheck['taskTime'] = dfDataCheck['taskTime_run1']
        elif 'Nimh-run2' in batchName:
            print('Making participant numbers in batch %s negative and -900000 to avoid overlap with MTurk subjects.'%batchName)
            dfDataCheck.participant = -dfDataCheck.participant-900000 # make negative to avoid overlap with MTurk participant numbers
            dfDataCheck['taskTime'] = dfDataCheck['taskTime_run2']
        elif 'Nimh-run3' in batchName:
            print('Making participant numbers in batch %s negative and -9900000 to avoid overlap with MTurk subjects.'%batchName)
            dfDataCheck.participant = -dfDataCheck.participant-9900000 # make negative to avoid overlap with MTurk participant numbers
            dfDataCheck['taskTime'] = dfDataCheck['taskTime_run3']

        # extract time of day for each subject
        for iSubj,subj in enumerate(dfDataCheck.participant):
            dfCoeffs.loc[dfCoeffs.Subject==subj,'TOD'] = dfDataCheck.loc[iSubj,'taskTime']


    # %% Plot results
    print('Plotting results...')
    # Set up figure
    plt.close(164)
    plt.figure(164,figsize=(9,4),dpi=120); plt.clf();

    # Intercept
    ax = plt.subplot(121)
    plt.plot(dfCoeffs.TOD,dfCoeffs['(Intercept)']*100,'.',label='particpant')
    isOk = pd.notna(dfCoeffs.TOD) & pd.notna(dfCoeffs['(Intercept)'])
    m,b = np.polyfit(dfCoeffs.TOD[isOk],dfCoeffs['(Intercept)'][isOk]*100, 1)
    x = np.array([np.min(dfCoeffs.TOD[isOk]),np.max(dfCoeffs.TOD[isOk])])
    plt.plot(x,m*x+b,label='Best fit')
    plt.xlabel('Time of day')
    plt.ylabel('Intercept parameter\n(% mood)')
    plt.xticks(np.arange(0,1.01,0.25))
    ax.set_xticklabels(['12am','6am','12pm','6pm','12am'])
    plt.grid(True)
    r,p = stats.spearmanr(dfCoeffs.TOD,dfCoeffs['(Intercept)'],nan_policy='omit')
    plt.title('Time of day vs. intercept parameter\n' + r'$r_s=%.3g, p_s=%.3g$'%(r,p))

    # Slope
    ax = plt.subplot(122)
    plt.plot(dfCoeffs.TOD,dfCoeffs.Time*100,'.',label='particpant')
    isOk = pd.notna(dfCoeffs.TOD) & pd.notna(dfCoeffs['Time'])
    nSubj = np.sum(isOk)
    m,b = np.polyfit(dfCoeffs.TOD[isOk],dfCoeffs['Time'][isOk]*100, 1)
    x = np.array([np.min(dfCoeffs.TOD[isOk]),np.max(dfCoeffs.TOD[isOk])])
    plt.plot(x,m*x+b,label='Best fit')
    plt.xlabel('Time of day')
    plt.ylabel('Slope parameter\n(% mood/min)')
    plt.xticks(np.arange(0,1.01,0.25))
    ax.set_xticklabels(['12am','6am','12pm','6pm','12am'])
    plt.grid(True)
    r,p = stats.spearmanr(dfCoeffs.TOD,dfCoeffs['Time'],nan_policy='omit')
    plt.title('Time of day vs. slope parameter\n' + r'$r_s=%.3g, p=%.3g$'%(r,p))
    print('Time of Day vs. slope parameter: Spearman r = %.3g, p = %.3g'%(r,p))

    # Annotate figure
    plt.tight_layout(rect=[0,0,1,0.93])
    plt.suptitle('%s batch (n=%d)'%(bigBatch,nSubj))
    print('Time of Day vs. intercept parameter: Spearman r = %.3g, p = %.3g'%(r,p))
    # Save figure
    outFile = '%s/Mmi-%s_TimeOfDayVsSlopeAndIntercept.png'%(outFigDir,bigBatch)
    print('Saving figure as %s...'%outFile)
    plt.savefig(outFile)
    outFile = '%s/Mmi-%s_TimeOfDayVsSlopeAndIntercept.pdf'%(outFigDir,bigBatch)
    print('Saving figure as %s...'%outFile)
    plt.savefig(outFile)
    print('Done!')
