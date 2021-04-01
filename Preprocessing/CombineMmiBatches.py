#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
CombineMmiBatches.py

Combine MTurk batches from the MMI experiment.

Created 6/5/20 by DJ.
Updated 3/31/21 by DJ - adapted for shared code structure.
"""
import pandas as pd
import numpy as np

def CombineMmiBatches(oldBatches,newBatch='',makeSubjectsMatchPymer=False,dataDir = '../Data/OutFiles'):
    """
    dfRating,dfTrial,dfSurvey,dfLifeHappy,pymerCoeffs,pymerInput = CombineMmiBatches(oldBatches,newBatch='',makeSubjectsMatchPymer=False,dataDir = '../Data/OutFiles')

    INPUTS:
    - oldBatches is a list of strings indicating the names of the batches you
      want to combine
    - newBatch is a string indicating the name of the batch you want to save.
      If this is empty or not provided, nothing will be saved.
    - makeSubjectsMatchPymer is a boolean indicating whether you want to make
      the subject numbers match what's used in RunPymerOnCovidData_Aug2020.
    - dataDir is a string indicating where the processed data files sit.

    OUTPUTS:
    - dfRating,dfTrial,dfSurvey, and dfLifeHappy are pandas dataframes with
      all batches combined. Each has an added column called 'batchName'.

    """

    # Set up
    dfRating_list = []
    dfTrial_list = []
    dfSurvey_list = []
    dfLifeHappy_list = []
    dfPymerCoeffs_list = []
    dfPymerInput_list = []
    # Load info from each batch and concatenate
    for batchName in oldBatches:
        # load info using standardized batch file names
        dfRating = pd.read_csv('%s/Mmi-%s_Ratings.csv'%(dataDir,batchName),index_col=0)
        dfTrial = pd.read_csv('%s/Mmi-%s_Trial.csv'%(dataDir,batchName),index_col=0)
        dfSurvey = pd.read_csv('%s/Mmi-%s_Survey.csv'%(dataDir,batchName),index_col=0)
        dfLifeHappy = pd.read_csv('%s/Mmi-%s_LifeHappy.csv'%(dataDir,batchName),index_col=0)
        try:
            dfPymerCoeffs = pd.read_csv('%s/Mmi-%s_pymerCoeffs.csv'%(dataDir,batchName))
            dfPymerInput = pd.read_csv('%s/Mmi-%s_pymerInput.csv'%(dataDir,batchName),index_col=0)
        except:
            participants = np.unique(dfRating.participant)
            dfPymerCoeffs = pd.DataFrame(np.ones((len(participants),2))*np.nan,columns = ['(Intercept)','Time'],index=participants)
            dfPymerInput = pd.DataFrame(np.ones((0,4))*np.nan,columns = ['Cohort','Subject','Mood','Time'])
        # add batch columns
        dfRating['batchName'] = batchName
        dfTrial['batchName'] = batchName
        dfSurvey['batchName'] = batchName
        dfLifeHappy['batchName'] = batchName
        dfPymerCoeffs['batchName'] = batchName
        dfPymerInput['batchName'] = batchName

        # Change subject numbers
        if makeSubjectsMatchPymer:
            if 'Nimh-run1' in batchName:
                # make negative to avoid overlap with MTurk participant numbers
                print('Making participant numbers in batch %s negative to avoid overlap with MTurk subjects.'%batchName)
                dfRating.participant = -dfRating.participant
                dfTrial.participant = -dfTrial.participant
                dfSurvey.participant = -dfSurvey.participant
                dfLifeHappy.participant = -dfLifeHappy.participant
                dfPymerCoeffs.Subject = -dfPymerCoeffs.Subject
                dfPymerInput.Subject = -dfPymerInput.Subject
            elif 'Nimh-run2' in batchName:
                # make negative to avoid overlap with MTurk participant numbers
                print('Making participant numbers in batch %s negative and -900000 to avoid overlap with MTurk subjects.'%batchName)
                dfRating.participant = -dfRating.participant - 900000
                dfTrial.participant = -dfTrial.participant - 900000
                dfSurvey.participant = -dfSurvey.participant - 900000
                dfLifeHappy.participant = -dfLifeHappy.participant - 900000
                dfPymerCoeffs.Subject = -dfPymerCoeffs.Subject - 900000
                dfPymerInput.Subject = -dfPymerInput.Subject - 900000

            elif 'Nimh-run3' in batchName:
                print('Making participant numbers in batch %s negative and -9900000 to avoid overlap with MTurk subjects.'%batchName)
                dfRating.participant = -dfRating.participant - 9900000
                dfTrial.participant = -dfTrial.participant - 9900000
                dfSurvey.participant = -dfSurvey.participant - 9900000
                dfLifeHappy.participant = -dfLifeHappy.participant - 9900000
                dfPymerCoeffs.Subject = -dfPymerCoeffs.Subject - 9900000
                dfPymerInput.Subject = -dfPymerInput.Subject - 9900000

            dfRating['participant']

        # add to list of dataframes
        dfRating_list.append(dfRating)
        dfTrial_list.append(dfTrial)
        dfSurvey_list.append(dfSurvey)
        dfLifeHappy_list.append(dfLifeHappy)
        dfPymerCoeffs_list.append(dfPymerCoeffs)
        dfPymerInput_list.append(dfPymerInput)

    # Concatenate across batches
    dfRating = pd.concat(dfRating_list,axis=0,ignore_index=True)
    dfTrial = pd.concat(dfTrial_list,axis=0,ignore_index=True)
    dfSurvey = pd.concat(dfSurvey_list,axis=0,ignore_index=True)
    dfLifeHappy = pd.concat(dfLifeHappy_list,axis=0,ignore_index=True)
    dfPymerCoeffs = pd.concat(dfPymerCoeffs_list,axis=0) # index is participant, so don't ignore
    dfPymerInput = pd.concat(dfPymerInput_list,axis=0,ignore_index=True)

    # Save results
    if newBatch!='':
        print('Saving new batch as %s...'%newBatch)
        dfRating.to_csv('%s/Mmi-%s_Ratings.csv'%(dataDir,newBatch))
        dfTrial.to_csv('%s/Mmi-%s_Trial.csv'%(dataDir,newBatch))
        dfSurvey.to_csv('%s/Mmi-%s_Survey.csv'%(dataDir,newBatch))
        dfLifeHappy.to_csv('%s/Mmi-%s_LifeHappy.csv'%(dataDir,newBatch))
        dfPymerCoeffs.to_csv('%s/Mmi-%s_pymerCoeffs.csv'%(dataDir,newBatch),index_label='Subject')
        dfPymerInput.to_csv('%s/Mmi-%s_pymerInput.csv'%(dataDir,newBatch))

    # Return results
    return dfRating,dfTrial,dfSurvey,dfLifeHappy,dfPymerCoeffs,dfPymerInput
