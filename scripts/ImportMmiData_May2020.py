#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
ImportMmiData_May2020.py

Import all adult online subjects (those recruited on MTurk) for the
Passage-of-Time Dysphoria Paper (Jangraw, 2021).

To use, modify batchNames in Cell 2 to import only certain batches (i.e.,
cohorts), or run without modifying to import all batches collected on MTurk.

- Created 4/21/20 by DJ.
- Updated 3/31/21 by DJ - adapted for shared code structure.
"""

# %% Import packages
from PassageOfTimeDysphoria.Preprocessing.MatchMTurkBatchToFiles import MatchMTurkBatchToFiles
from PassageOfTimeDysphoria.Preprocessing.GetMmiRatingsAndTimes import GetMmiRatingsAndTimes
from PassageOfTimeDysphoria.Preprocessing.ImportMmiSurveyData import ImportMmiSurveyData
from PassageOfTimeDysphoria.Preprocessing.ScoreMmiSurvey import ScoreMmiSurvey
import PassageOfTimeDysphoria.Analysis.PlotMmiData as pmd
import numpy as np
import pandas as pd
import os.path
from matplotlib import pyplot as plt

# %% Declare constants and batches to be imported
# Declare batches to be imported (see Mmi-Batches.csv batchName col for options)
batchNames = ['COVID01','COVID02','COVID03','Expectation-7min','Expectation-12min',
              'Motion','MotionFeedback','Numbers','Recovery1','RecoveryInstructed1',
              'RecoveryInstructed1Freq0p5','RecoveryInstructed1Freq0p25',
              'RecoveryInstructed1Freq2','RestDownUp', 'Return1',
              'Stability01-RandomVer2','Stability02-RandomVer2',
              'Stability01-Rest','Stability02-Rest',
              'Stability01-Closed','Stability02-Closed']
dataCheckDir = '../Data/DataChecks' # where dataCheck files sit
rawDataDir = '../Data/PilotData' # where raw data files sit
dataDir = '../Data/OutFiles' # where processed data sits
outFigDir = '../Figures' # where results should go
plotEveryParticipant = False # should we make a plot for every participant?
overwrite = False; # overwrite previous results if they already exist?


# %% Import data
for batchName in batchNames:
    pilotDataFolder = '%s/%s'%(rawDataDir,batchName)
    batchFile = '%s/Batch_%s_results.csv'%(rawDataDir,batchName)

    if batchName.startswith('COVID'):
        demoDataFile='%s/COVID01_DataCheck.csv'%dataCheckDir
    elif batchName.startswith('Stability02'):
        condition = batchName.split('-')[1]
        demoDataFile = '%s/Stability01-%s_DataCheck.csv'%(dataCheckDir,condition)
    else:
        demoDataFile = ''

    # Import data check file
    dataCheckFile = '%s/%s_DataCheck.csv'%(dataCheckDir,batchName)

    if os.path.exists(dataCheckFile) and not overwrite:
        # read dataCheck
        print('==== Reading data check from %s...'%dataCheckFile)
        dfDataCheck = pd.read_csv(dataCheckFile,index_col=0);
    else:
        print('==== Checking data from batch file %s...'%batchFile)
        dfDataCheck = MatchMTurkBatchToFiles(batchFile,pilotDataFolder,batchName,demoDataFile)
        # Save dataCheck
        print('==== Saving data check as %s...'%dataCheckFile)
        dfDataCheck.to_csv(dataCheckFile)

    print('==== Done!')


    # %% Import task & survey files from all complete subjects

    nComplete = np.sum(dfDataCheck['isComplete'])
    print('=== %d/%d subjects complete. ==='%(nComplete,dfDataCheck.shape[0]))
    dfDataCheck_complete = dfDataCheck.loc[dfDataCheck.isComplete,:].reset_index(drop=True)

    surveyList = []
    trialList = []
    ratingList =[]
    lifeHappyList = []
    nSubj = dfDataCheck_complete.shape[0]
    for iLine in range(nSubj):
        # Print status
        print('=== Importing Subject %d/%d... ==='%(iLine,nSubj))
        # Task
        inFile = dfDataCheck_complete.loc[iLine,'taskFile']
        inFile = inFile.replace('../PilotData',rawDataDir) # replace relative path from data check file with relative path from here
        dfTrial,dfRating,dfLifeHappy = GetMmiRatingsAndTimes(inFile)
        # Get summary of task data
        trialList.append(dfTrial)
        ratingList.append(dfRating)
        lifeHappyList.append(dfLifeHappy)

        # Survey
        inFile = dfDataCheck_complete.loc[iLine,'surveyFile']
        inFile = inFile.replace('../PilotData',rawDataDir) # replace relative path from data check file with relative path from here
        mTurkID = dfDataCheck_complete.loc[iLine,'MTurkID']
        dfQandA = ImportMmiSurveyData(inFile, mTurkID, demoDataFile)
        # Score survey and add to survey list
        participant = dfDataCheck_complete.loc[iLine,'participant']
        surveyList.append(ScoreMmiSurvey(dfQandA,dfDataCheck_complete.loc[iLine,'participant']))

        # Plot task data
        if plotEveryParticipant:
            plt.figure(1,figsize=(10,4),dpi=180, facecolor='w', edgecolor='k');
            plt.clf();
            ax1 = plt.subplot(2,1,1);
            pmd.PlotMmiRatings(dfTrial,dfRating,'line')
            ax2 = plt.subplot(2,1,2);
            plt.xlim(ax1.get_xlim())
            pmd.PlotMmiRPEs(dfTrial,dfRating)
            plt.tight_layout()
            # Save figure
            outFig = '%s/Mmi-%s-%s.png'%(outFigDir,batchName,participant)
            print('Saving figure as %s...'%outFig)
            plt.savefig(outFig)

    print('=== Done! ===')
    # %% Append across lists
    dfTrial = pd.concat(trialList);
    dfRating = pd.concat(ratingList);
    dfSurvey = pd.concat(surveyList);
    dfLifeHappy = pd.concat(lifeHappyList);

    # %% Save?
    files = {'trial': '%s/Mmi-%s_Trial.csv'%(dataDir,batchName),
             'ratings': '%s/Mmi-%s_Ratings.csv'%(dataDir,batchName),
             'survey':'%s/Mmi-%s_Survey.csv'%(dataDir,batchName),
             'lifeHappy':'%s/Mmi-%s_LifeHappy.csv'%(dataDir,batchName)}

    tables = {'trial': dfTrial,
             'ratings': dfRating,
             'survey': dfSurvey,
             'lifeHappy': dfLifeHappy}

    for item in [x[0] for x in files.items()]:
        if os.path.exists(files[item]) and not overwrite:
            # read dataCheck
            print('==== Reading %s from %s...'%(item,files[item]))
            tables[item] = pd.read_csv(files[item],index_col=0);
        else:
            # Save dataCheck
            print('==== Saving %s as %s...'%(item,files[item]))
            tables[item].to_csv(files[item])

    print('==== Done!')

    # %% Plot averages
    dfRatingMean = pmd.GetMeanRatings(dfRating)
    dfTrialMean = pmd.GetMeanTrials(dfTrial)

    # Plot results
    plt.figure(1,figsize=(10,4),dpi=180, facecolor='w', edgecolor='k');
    plt.clf();
    ax1 = plt.subplot(2,1,1);
    pmd.PlotMmiRatings(dfTrialMean,dfRatingMean,'line',autoYlim=True)
    plt.title('MMI %s mean (n=%d)'%(batchName,nSubj))
    ax2 = plt.subplot(2,1,2);
    plt.xlim(ax1.get_xlim())
    pmd.PlotMmiRPEs(dfTrialMean,dfRatingMean)
    plt.title('MMI %s mean (n=%d)'%(batchName,nSubj))
    plt.tight_layout()
    # Save figure
    outFig = '%s/Mmi-%s-MeanRatings.png'%(outFigDir,batchName)
    print('Saving figure as %s...'%outFig)
    plt.savefig(outFig)
    print('Done!')


    # %% Generate keys matching two datasets

    # find matches in a dataCheck file
    if not demoDataFile=='':
        dfDataCheck01 = pd.read_csv(demoDataFile, index_col=0)
        for iSubj in dfDataCheck.index:
            # add key of original participant number
            isMatch = (dfDataCheck01['MTurkID'] == dfDataCheck.loc[iSubj,'MTurkID'])
            dfDataCheck.loc[iSubj,'participant_day1'] = dfDataCheck01.loc[isMatch,'participant'].values[0]
            dfDataCheck.loc[iSubj,'participant_day2'] = dfDataCheck.loc[iSubj,'participant']
            try:
                dfDataCheck.loc[iSubj,'taskFile_day1'] = os.path.basename(dfDataCheck01.loc[isMatch,'taskFile'].values[0])
            except:
                dfDataCheck.loc[iSubj,'taskFile_day1'] = np.nan;
            try:
                dfDataCheck.loc[iSubj,'taskFile_day2']= os.path.basename(dfDataCheck.loc[iSubj,'taskFile'])
            except:
                dfDataCheck.loc[iSubj,'taskFile_day2'] = np.nan;

        colsToKeep = ['participant_day1','participant_day2','taskFile_day1','taskFile_day2','location','gender','age','status']
        dfKey = dfDataCheck.loc[:,colsToKeep]
        outFile = '%s/%s_key.xlsx'%(dataDir,batchName)
        print('Saving key to %s...'%outFile)
        if os.path.exists(outFile) and not overwrite:
            print('Not overwriting.')
        else:
            dfKey.to_excel(outFile)
