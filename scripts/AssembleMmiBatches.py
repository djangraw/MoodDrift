#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AssembleMmiBatches.py

Gathers all the batches imported by ImportMmiData_May2020.py, gets info about
them, and saves that info to table Mmi-Batches.csv. Also combines groups of
batches that will later be analyzed as a single group.

Created 6/2/20 by DJ.
Updated 3/31/21 by DJ - adapted for shared code structure.
Updated 4/2/21 by DJ - added overwrite flag.
Updated 4/8/21 by DJ - added line to calculate Recovery(Instructed)1.
"""

# %% Import packages
import pandas as pd
import numpy as np
import os
from glob import glob
import PassageOfTimeDysphoria.Analysis.PlotMmiData as pmd
from PassageOfTimeDysphoria.Preprocessing.CombineMmiBatches import CombineMmiBatches

# Get batch names
dataCheckDir = '../Data/DataChecks'
outDir = '../Data/OutFiles'
includeRepeats = False; # should the "AllOpeningRestAndRandom" superbatch include returning subjects?
overwrite = True # overwrite existing files?

print('Gathring batches...')
batchFiles_glob = glob('%s/*DataCheck.csv'%dataCheckDir)
batchFiles_glob.sort()
print('%d batches found.'%len(batchFiles_glob))

batchFiles = []
batchNames = []
batchStart = []
batchEnd = []
taskFiles = []
nSubjAttempted = []
nSubjCompleted = []
# %%
print('Getting info from DataCheck files...')
for batchFile in batchFiles_glob:
    print('   batch %s...'%batchFile)
    dfDataCheck = pd.read_csv(batchFile,index_col=0) # read in file
    batchName = os.path.basename(batchFile).split('_')[0] # extract name of batch

    # Online adolescent participants: split across runs
    if 'Nimh' in batchFile:
        nRuns = 3
        for run in range(1,nRuns+1):
            minDate = '2999-01-01'
            maxDate = '1999-01-01'
            for iLine in range(dfDataCheck.shape[0]):
                taskFile = dfDataCheck.loc[iLine,'taskFile_run%d'%run]
                if isinstance(taskFile,str) and not taskFile[0].isdigit(): # if it's a string and not '0.0'
                    minDate = min(minDate,dfDataCheck.loc[iLine,'taskFile_run%d'%run].split('_')[-2])
                    maxDate = max(maxDate,dfDataCheck.loc[iLine,'taskFile_run%d'%run].split('_')[-2])
            batchNames.append('%s-run%s'%(batchName,run))
            batchFiles.append(batchFile)
            batchStart.append(minDate)
            batchEnd.append(maxDate)
            # Add number of subjects
            nSubjAttempted.append(np.sum(pd.notna(dfDataCheck['taskFile_run%d'%run])))
            nSubjCompleted.append(np.sum(pd.notna(dfDataCheck['taskFile_run%d'%run])))
    else: # online adult participants: no splitting
        minDate = '2999-01-01'
        maxDate = '1999-01-01'
        for iLine in range(dfDataCheck.shape[0]):
            taskFile = dfDataCheck.loc[iLine,'taskFile']
            if isinstance(taskFile,str) and not taskFile[0].isdigit(): # if it's a string and not '0.0'
                minDate = min(minDate,dfDataCheck.loc[iLine,'taskFile'].split('_')[-2])
                maxDate = max(maxDate,dfDataCheck.loc[iLine,'taskFile'].split('_')[-2])
        batchNames.append(batchName)
        batchFiles.append(batchFile)
        batchStart.append(minDate)
        batchEnd.append(maxDate)
        # Add number of subjects
        nSubjAttempted.append(dfDataCheck.shape[0])
        nSubjCompleted.append(np.sum(dfDataCheck.isComplete))

# Create Dataframe
print('Creating dataframe...')
maxNBlocks = 4
cols = ['batchName','startDate','endDate','nSubjAttempted','nSubjCompleted',
        'dataCheckFile','ratingsFile','trialFile','surveyFile','lifeHappyFile',
        'pymerInputFile','pymerCoeffsFile','nPreviousRuns','isNimhCohort'] + \
        ['block%d_type'%run for run in range(maxNBlocks)] + \
        ['block%d_targetHappiness'%run for run in range(maxNBlocks)] + \
        ['block%d_nTrials'%run for run in range(maxNBlocks)] + \
        ['block%d_nRatings'%run for run in range(maxNBlocks)] + \
        ['block%d_meanDuration'%run for run in range(maxNBlocks)]

nBatches = len(batchNames)
dfBatches = pd.DataFrame(np.ones((nBatches,len(cols)))*np.nan,columns=cols)
dfBatches['batchName'] = batchNames
dfBatches['startDate'] = batchStart
dfBatches['endDate'] = batchEnd
dfBatches['nSubjAttempted'] = nSubjAttempted
dfBatches['nSubjCompleted'] = nSubjCompleted
dfBatches['dataCheckFile'] = batchFiles
dfBatches['ratingsFile'] = ['%s/Mmi-%s_Ratings.csv'%(outDir,batchName) for batchName in batchNames]
dfBatches['trialFile'] = ['%s/Mmi-%s_Trial.csv'%(outDir,batchName) for batchName in batchNames]
dfBatches['surveyFile'] = ['%s/Mmi-%s_Survey.csv'%(outDir,batchName) for batchName in batchNames]
dfBatches['lifeHappyFile'] = ['%s/Mmi-%s_LifeHappy.csv'%(outDir,batchName) for batchName in batchNames]
dfBatches['pymerInputFile'] = ['%s/Mmi-%s_pymerInput.csv'%(outDir,batchName) for batchName in batchNames]
dfBatches['pymerCoeffsFile'] = ['%s/Mmi-%s_pymerCoeffs.csv'%(outDir,batchName) for batchName in batchNames]
dfBatches['nPreviousRuns'] = 0;
dfBatches.loc[['02' in batchName for batchName in batchNames],'nPreviousRuns'] = 1
dfBatches.loc[['run2' in batchName for batchName in batchNames],'nPreviousRuns'] = 1
dfBatches.loc[['03' in batchName for batchName in batchNames],'nPreviousRuns'] = 2
dfBatches.loc[['run3' in batchName for batchName in batchNames],'nPreviousRuns'] = 2
dfBatches['isNimhCohort'] = ['Nimh' in batchName for batchName in batchNames]

# Get block info
print('Getting block info...')
for iBatch,batchName in enumerate(batchNames):
    print('   batch %s...'%batchName)
    # Load
    dfRating = pd.read_csv('%s/Mmi-%s_Ratings.csv'%(outDir,batchName))
    dfTrial = pd.read_csv('%s/Mmi-%s_Trial.csv'%(outDir,batchName))

    # Get averages
    dfRatingMean = pmd.GetMeanRatings(dfRating)
    dfTrialMean = pmd.GetMeanTrials(dfTrial)

    tBlockSwitch,blockType = pmd.GetBlockTimes(dfTrial,dfRating)
    nBlocks = len(blockType)
    for iBlock in range(nBlocks):
        isThis = (dfTrialMean.iBlock==iBlock)
        dfBatches.loc[iBatch,'block%d_type'%iBlock] = dfTrialMean.loc[isThis,'trialType'].values[0]
        dfBatches.loc[iBatch,'block%d_targetHappiness'%iBlock] = dfTrialMean.loc[isThis,'targetHappiness'].values[0]
        dfBatches.loc[iBatch,'block%d_nTrials'%iBlock] = np.sum(isThis)
        isThis = (dfRatingMean.iBlock==iBlock)
        dfBatches.loc[iBatch,'block%d_nRatings'%iBlock] = np.sum(isThis)
        dfBatches.loc[iBatch,'block%d_meanDuration'%iBlock] = tBlockSwitch[iBlock+1] - tBlockSwitch[iBlock]

outFile = '%s/Mmi-Batches.csv'%outDir
print('Writing to %s...'%outFile)
if os.path.exists(outFile) and not overwrite:
    print('Not overwriting existing file.')
else:
    dfBatches.to_csv(outFile)
    print('Done!')

# %% Combine two batches with identical trials
    
CombineMmiBatches(['Recovery1','RecoveryInstructed1'],'Recovery(Instructed)1');

# %% Assemble batches with no opening rest, short opening rest, and long opening rest

# Reload batch info file
batchFile = '%s/Mmi-Batches.csv'%outDir
print('Reading batch info from %s...'%batchFile)
dfBatches = pd.read_csv(batchFile,index_col=0)

# Get batches matching our description
isNoRestBatch = (dfBatches.nPreviousRuns==0) & \
             (dfBatches.block0_type=='closed') & \
             ((dfBatches.block0_targetHappiness=='1.0') | \
              (dfBatches.block0_targetHappiness=='1'))
isShortBatch = (dfBatches.nPreviousRuns==0) & \
             (dfBatches.block0_type=='rest') & \
             (dfBatches.block1_type=='closed') & \
             ((dfBatches.block1_targetHappiness=='1.0') | \
              (dfBatches.block1_targetHappiness=='1')) & \
             (dfBatches.block0_meanDuration<500)
isLongBatch = (dfBatches.nPreviousRuns==0) & \
             (dfBatches.block0_type=='rest') & \
             (dfBatches.block1_type=='closed') & \
             ((dfBatches.block1_targetHappiness=='1.0') | \
              (dfBatches.block1_targetHappiness=='1')) & \
             (dfBatches.block0_meanDuration>=500)


# Create new "superbatches" that combine multiple cohorts
CombineMmiBatches(dfBatches.loc[isNoRestBatch,'batchName'].values,'NoOpeningRest');
CombineMmiBatches(dfBatches.loc[isShortBatch,'batchName'].values,'ShortOpeningRest');
CombineMmiBatches(dfBatches.loc[isLongBatch,'batchName'].values,'LongOpeningRest');
CombineMmiBatches(dfBatches.loc[isShortBatch | isLongBatch,'batchName'].values,'AnyOpeningRest');
CombineMmiBatches(dfBatches.loc[isNoRestBatch | isShortBatch | isLongBatch,'batchName'].values,'AnyOrNoRest');

# %% Assemble batch of all cohorts with opening rest or random-gambling block for use in large-scale LME analysis

# Get batches that match our description
dfBatch = dfBatches[['batchName','ratingsFile','surveyFile','trialFile','lifeHappyFile','block0_type','nPreviousRuns']]
dfBatch = dfBatch.loc[(dfBatches.block0_type=='rest') | (dfBatches.block0_type=='random'),:]
if not includeRepeats:
    dfBatch = dfBatch.loc[(dfBatch.nPreviousRuns==0),:]
dfBatch = dfBatch.drop(['Stability01-random','Stability02-random','RecoveryNimh-run3'],axis=0,errors='ignore')

# Create "superbatch" that combines multiple cohorts
CombineMmiBatches(dfBatch['batchName'].values,'AllOpeningRestAndRandom',makeSubjectsMatchPymer=True);
