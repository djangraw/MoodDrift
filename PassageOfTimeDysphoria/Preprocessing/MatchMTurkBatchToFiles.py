#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Get raw data files for an MTurk batch.

Created 4/21/20 by DJ.
Updated 3/31/21: removed outDir input
Updated 12/10/21 by DJ - make survey file detection caps-agnostic
"""

import pandas as pd
import numpy as np
import os.path
import glob
import dateutil.parser as dparser
from .ImportMmiSurveyData import ImportMmiSurveyData


def MatchMTurkBatchToFiles(batchFile,pilotDataFolder,batchName='',demoDataFile=''):
    # declare default batchName=(topmost directory of pilotDataFolder)
    if len(batchName)==0:
        batchName = os.path.split(pilotDataFolder)[-1]

    # %% Load batch file

    # check for file existing
    if not os.path.exists(batchFile):
        raise OSError('batch file %s not found!'%batchFile)

    # Load batch file
    dfBatch = pd.read_csv(batchFile);
    nSubj = dfBatch.shape[0];

    # Initialize data check dataframe
    demographics = ['location','gender','age','status']
    cols = ['MTurkID','batchDate','participant','taskCode','surveyCode','taskFile','surveyFile','catchCorrect','cashBonus'] + demographics
    batchDate = dparser.parse(dfBatch['CreationTime'][0],fuzzy=True);
    batchDateStr = '%04d-%02d-%02d'%(batchDate.year,batchDate.month,batchDate.day)
    dfDataCheck = pd.DataFrame(np.ones((nSubj,len(cols)))*np.nan,columns=cols)
    dfDataCheck['MTurkID'] = dfBatch['WorkerId']
    dfDataCheck['batchDate'] = batchDateStr
    dfDataCheck['participant'] = dfBatch['Input.participant']
    dfDataCheck['taskCode'] = dfBatch['Answer.taskcode']
    dfDataCheck['surveyCode'] = dfBatch['Answer.surveycode']
    # dfDataCheck.loc[:,['taskFile','surveyFile','catchCorrect']+demographics] = np.nan;
    dfDataCheck['cashBonus'] = 0;

    # Look for matching files
    uniqueParticipants = np.unique(dfBatch['Input.participant']);
    for iSubj,participant in enumerate(uniqueParticipants):
        allFiles = glob.glob('%s/%s_*.csv'%(pilotDataFolder,participant))
        print('== Participant %d/%d: %s'%(iSubj,uniqueParticipants.size,participant))
        for thisFile in allFiles:
            print('Checking %s...'%os.path.basename(thisFile))
            dfIn = pd.read_csv(thisFile);
            expName = dfIn.expName[0]
            if 'hashCode' in dfIn.columns:
                hashCode = dfIn.loc[pd.notna(dfIn.hashCode),'hashCode'].values[0]
                # convert to string
                if not isinstance(hashCode,str):
                    hashCode = '%d'%hashCode
                
                if 'survey' in expName.lower(): # it's a survey
                    fileType = 'survey'
                else:
                    fileType = 'task'
                # find a match
                isThisSubj = (dfDataCheck['%sCode'%fileType].astype(str)==hashCode) & (dfDataCheck['participant']==participant)
                if np.sum(isThisSubj)==1:
                    # If we found a batch file line matching this file...
                    if np.isnan(dfDataCheck.loc[isThisSubj,'%sFile'%fileType].values[0]): # nothing there yet
                        dfDataCheck.loc[isThisSubj,'%sFile'%fileType] = thisFile;
                        # Add in other info
                        if fileType=='survey':
                            dfQandA = ImportMmiSurveyData(thisFile,dfDataCheck.loc[isThisSubj,'MTurkID'].values[0], demoDataFile)
                            # number of catch q's correct
                            isCatch = dfQandA.Survey=='CATCH'
                            catchCorrect = np.sum(dfQandA.loc[isCatch,'iAnswer']==dfQandA.loc[isCatch,'iCatchAnswer'])
                            dfDataCheck.loc[isThisSubj,'catchCorrect'] = catchCorrect
                            # demographics
                            for demo in demographics:
                                dfDataCheck.loc[isThisSubj,demo] = dfQandA.loc[dfQandA.Question==demo,'Answer'].values[0]
                        else:
                            dfDataCheck.loc[isThisSubj,'cashBonus'] = dfIn.loc[pd.notna(dfIn['cashBonus']),'cashBonus'].values[0]
                    else: # something was already there
                        oldFile = os.path.basename(dfDataCheck.loc[isThisSubj,'surveyFile'].values[0])
                        newFile = os.path.basename(thisFile);
                        raise ValueError('Files %s and %s both match %sCode %s! Choose one to delete.'%(oldFile,newFile,fileType,hashCode));
                elif np.sum(isThisSubj)==0:
                    print('No lines in batch file matching %sCode=%s.'%(fileType,hashCode))
                else:
                    raise ValueError('Expected 1 batch line matching %sCode=%s... %d found!'%(fileType,hashCode,np.sum(isThisSubj)))
            else:
                print('File %s has no hashCode recorded!'%os.path.basename(thisFile))

    dfDataCheck['isComplete'] = pd.notna(dfDataCheck.taskFile) & pd.notna(dfDataCheck.surveyFile);



    return dfDataCheck
