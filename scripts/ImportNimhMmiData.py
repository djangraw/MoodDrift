#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ImportNimhMmiData.py

Import MMI data from in-person NIMH subjects performing the task online during
the COVID-19 pandemic.

Created on Tue May 19 13:41:11 2020

@author: jangrawdc
- Updated 6/2/20 by DJ - renamed dfDataCheck
- Updated 3/31/21 by DJ - adapted for shared code structure.
- Updated 4/2/21 by DJ - allowed date strings in MM/DD/YY format, use and accommodate de-ID'ed demographics file.
- Updated 12/10/21 by DJ - added dfProbe output to GetMmiRatingsAndTimes.
"""

# %% Set up
# Import packages
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import glob
import dateutil.parser as dparser
import datetime
import PassageOfTimeDysphoria.Analysis.PlotMmiData as pmd
from PassageOfTimeDysphoria.Preprocessing.GetMmiRatingsAndTimes import GetMmiRatingsAndTimes
import os.path

# Convert MM/DD/YY datestr to year, month, and day ints
def GetYearMonthDay(dateStr):
    if '-' in dateStr: #YYYY-MM-DD format
        dateYear,dateMonth,dateDay = [int(x) for x in dateStr.split('-')]
    elif '/' in dateStr: # MM/DD/YY format
        dateMonth,dateDay,dateYear2 = [int(x) for x in dateStr.split('/')]
        # convert to 4-digit year
        if dateYear2<21:
            dateYear = 2000+dateYear2
        else:
            dateYear = 1900+dateYear2
    else:
        raise ValueError('date string must be in YYYY-MM-DD or MM/DD/YY format')

    return dateYear,dateMonth,dateDay

# Get age from datestrings for current time and DOB
def AgeFromDateStrings(currStr,dobStr):
    # Convert stting to day, month, and year
    currYear,currMonth,currDay = GetYearMonthDay(currStr)
    dobYear,dobMonth,dobDay = GetYearMonthDay(dobStr)
    # convert to datetime objects
    currDatetime = datetime.datetime(currYear,currMonth,currDay)
    dobDatetime = datetime.datetime(dobYear,dobMonth,dobDay)
    # get difference, convert to years
    ageInYears = (currDatetime-dobDatetime).days/365.25 # get age in years (days/365.25 for leap year)
    # return result
    return ageInYears

# %% Load demographics data
# Import data
rawDataDir = '../Data/PilotData' # where pilot data can be found
dataCheckDir = '../Data/DataChecks' # where data check files should be saved
procDataDir = '../Data/OutFiles' # where preprocessed data should be saved
outFigDir = '../Figures' # where figures should be saved
batchName = 'RecoveryNimh' # Name of this batch
plotEveryParticipant = False # should we make a plot for every participant?
overwrite = True; # overwrite previous results if they already exist?

#demoFile = '%s/MmiRecoveryNimh_Demographics.csv'%rawDataDir # Demographics file for NIMH participants including DOB
demoFile = '%s/MmiRecoveryNimh_Demographics_nodob.csv'%rawDataDir # deID's Demographics file for NIMH participants in which DOBs are replaced with floored age

# Load demographics data
print('=== Reading and cropping %s...'%demoFile)
dfData = pd.read_csv(demoFile)

if 'AllData' in demoFile:
    # extract info for each subject
    cols = ['participant','SEX','DOB','Participant_Type','Age']
    dfSurvey = dfData.loc[:,cols].drop_duplicates().reset_index(drop=True)
    # adjust to match MTurk names/values
    dfSurvey.loc[dfSurvey.SEX=='MALE','SEX'] = 'Male'
    dfSurvey.loc[dfSurvey.SEX=='FEMALE','SEX'] = 'Female'
    dfSurvey = dfSurvey.rename(columns={'SEX':'gender',
                            'Age':'age',
                            'Participant_Type':'diagnosis'})

elif 'Demographics' in demoFile:
    # extract info for each subject
    isBase = pd.notna(dfData.s_crisis_base_date)
    dfData.loc[isBase,['s_crisis_fu_tot','s_crisis_fu_date']] = dfData.loc[isBase,['s_crisis_base_tot','s_crisis_base_date']]
    try:
        cols = ['participant','SEX','DOB','Participant_Type','s_crisis_fu_tot','s_mfq_tot','s_scaredshort_tot','s_crisis_fu_date','age']
        dfSurvey = dfData.loc[:,cols].drop_duplicates(cols[:-1]).reset_index(drop=True) # drop based on everything except age
    except KeyError:
        cols = ['participant','SEX','Participant_Type','s_crisis_fu_tot','s_mfq_tot','s_scaredshort_tot','s_crisis_fu_date','age']
        dfSurvey = dfData.loc[:,cols].drop_duplicates(cols[:-1]).reset_index(drop=True) # drop based on everything except age
    # dfSurvey['age'] = np.nan
    # adjust to match MTurk names/values
    dfSurvey.loc[dfSurvey.SEX=='MALE','SEX'] = 'Male'
    dfSurvey.loc[dfSurvey.SEX=='FEMALE','SEX'] = 'Female'
    dfSurvey = dfSurvey.rename(columns={'SEX':'gender',
                            'Participant_Type':'diagnosis',
                            's_crisis_fu_tot':'CRISIS',
                            's_mfq_tot':'MFQ',
                            's_scaredshort_tot':'SCARED',
                            's_crisis_fu_date': 'DateOfSurvey'})
    # Crop to COMPLETE measurements only
    nSubj_orig = np.unique(dfSurvey.participant).size
    isOk = pd.notna(dfSurvey.CRISIS) & pd.notna(dfSurvey.MFQ) & pd.notna(dfSurvey.SCARED)
    dfSurvey = dfSurvey.loc[isOk,:]

    # Crop to the ***FIRST measurement*** for each subject
    participants = np.unique(dfSurvey.participant)
    nDupes = 0;
    for participant in participants:
        isThis = dfSurvey.participant==participant
        if np.sum(isThis)>1:
            earliestDate = np.min(dfSurvey.loc[isThis,'DateOfSurvey'])
            isNotEarliest = isThis & (dfSurvey.DateOfSurvey>earliestDate)
            nDupes = nDupes + np.sum(isNotEarliest)
            dfSurvey = dfSurvey.drop(isNotEarliest[isNotEarliest].index,axis=0)
#            print(' - subj %d: %d duplicates.'%(participant,np.sum(isNotEarliest)))
    dfSurvey = dfSurvey.reset_index(drop=True)
    print('Deleted %d duplicate lines.'%(nDupes))


participants = np.unique(dfSurvey.participant)
nSubj = participants.size
assert (dfSurvey.shape[0]==nSubj) # make sure all participants are unique
print('%d of %d subjects had complete survey data.'%(nSubj,nSubj_orig))

# Save results
#outFile = '%s/Mmi-%s_Survey.csv'%(procDataDir,batchName)
#print('Saving to %s...'%outFile)
#if os.path.exists(outFile) and not overwrite:
#    print('Not overwriting existing file.')
#else:
#    dfSurvey.to_csv(outFile)
#    print('Done!')


# %% Load data from Pavlovia

# data check file is distributed with shared data
dataCheckFile = '%s/%s_DataCheck.csv'%(dataCheckDir,batchName)
dfDataCheck = pd.read_csv(dataCheckFile)
maxNRuns = 3

# Old code to create datacheck file
#dfDataCheck = pd.DataFrame(participants,columns=['participant'])
##dfDataCheck['taskFile'] = '';
#maxNRuns = 0;
#for iSubj,participant in enumerate(participants):
#    # get list of files from this participant
#    files = np.array(glob.glob('%s/%s/%s*.csv'%(rawDataDir,batchName,participant)))
#    # get completion indicator and date
#    fileDate = np.zeros(len(files),datetime.datetime)
#    isComplete = np.zeros(len(files),bool)
#    for iFile,thisFile in enumerate(files):
#        dfIn = pd.read_csv(thisFile);
#        fileDate[iFile] = dparser.parse(dfIn['date'][0].split('_')[0],fuzzy=True);
#        isComplete[iFile] = ('cashBonus' in dfIn.columns)
#    # crop
#    files = files[isComplete]
#    fileDate = fileDate[isComplete]
#    # sort by date
#    iSorted = np.argsort(fileDate)
#    files = files[iSorted]
#
#    # add files to dfDataCheck dataframe
#    for iFile,thisFile in enumerate(files):
#        dfDataCheck.loc[iSubj,'taskFile_run%d'%(iFile+1)] = files[iFile]
#    maxNRuns = max(maxNRuns,len(files))
#    if len(files)==0:
#        print('***WARNING: participant %d has %d complete data files!'%(participant,len(files)))
#
#outFile = '%s/%s_DataCheck.csv'%(dataCheckDir,batchName)
#print('Saving to %s...'%outFile)
#if os.path.exists(outFile) and not overwrite:
#    print('Not overwriting existing file.')
#else:
#    dfDataCheck.to_csv(outFile)
#    print('Done!')

# %% Import data
dfDataCheck['isComplete'] = True;
nComplete = np.sum(dfDataCheck['isComplete'])
print('=== %d/%d subjects complete. ==='%(nComplete,dfDataCheck.shape[0]))
dfDataCheck_complete = dfDataCheck.loc[dfDataCheck.isComplete,:].reset_index(drop=True)

trialList = []
ratingList =[]
lifeHappyList = []
nSubj = dfDataCheck_complete.shape[0]
for iLine in range(nSubj):
    # Print status
    print('=== Importing Subject %d/%d... ==='%(iLine,nSubj))
    participant = dfDataCheck.loc[iLine,'participant']

    for iRun in range(maxNRuns):
        run = iRun+1
        if isinstance(dfDataCheck_complete.loc[iLine,'taskFile_run%d'%run],str): # if it's a string
            date = dfDataCheck_complete.loc[iLine,'taskFile_run%d'%run].split('_')[2]
            # Task
            inFile = dfDataCheck_complete.loc[iLine,'taskFile_run%d'%run]
            inFile = inFile.replace('../PilotData',rawDataDir) # replace relative path from data check file with relative path from here
            dfTrial,dfRating,dfLifeHappy,dfProbe = GetMmiRatingsAndTimes(inFile)
            # Add run/date info
            for df in [dfTrial,dfRating,dfLifeHappy]:
                df['run'] = run
                df['date'] = date
            # append to lists
            trialList.append(dfTrial)
            ratingList.append(dfRating)
            lifeHappyList.append(dfLifeHappy)

            # Plot task data
            if plotEveryParticipant:
                plt.figure(1,figsize=(10,4),dpi=180, facecolor='w', edgecolor='k');
                plt.clf();
                ax1 = plt.subplot(2,1,1);
                pmd.PlotMmiRatings(dfTrial,dfRating,'line')
                plt.title('MMI participant %d, run %d'%(participant,run))
                ax2 = plt.subplot(2,1,2);
                plt.xlim(ax1.get_xlim())
                pmd.PlotMmiRPEs(dfTrial,dfRating)
                plt.tight_layout()
                # Save figure
                outFig = '%s/Mmi-%s-%s-run%d.png'%(outFigDir,batchName,participant,run)
                print('Saving figure as %s...'%outFig)
                plt.savefig(outFig)

print('=== Done! ===')
# %% Append across lists
dfTrial = pd.concat(trialList);
dfRating = pd.concat(ratingList);
dfLifeHappy = pd.concat(lifeHappyList);

# %% Save?

files = {'trial': '%s/Mmi-%s_Trial.csv'%(procDataDir,batchName),
         'ratings': '%s/Mmi-%s_Ratings.csv'%(procDataDir,batchName),
         'lifeHappy':'%s/Mmi-%s_LifeHappy.csv'%(procDataDir,batchName)}

tables = {'trial': dfTrial,
         'ratings': dfRating,
         'lifeHappy': dfLifeHappy}

for item in [x[0] for x in files.items()]:
    if os.path.exists(files[item]) and not overwrite:
        # read dataCheck
        print('==== Reading %s from %s...'%(item,files[item]))
        tables[item] = pd.read_csv(files[item],index_col=0);
    else:
        # Save csv file
        print('==== Saving %s as %s...'%(item,files[item]))
        tables[item].to_csv(files[item])
        # save csv file for single run
        for iRun in range(maxNRuns):
            run = iRun+1
            dfRun = tables[item].loc[tables[item].run==run,:]
            outFile = files[item].replace(batchName,'%s-run%d'%(batchName,run))
            print('==== Saving %s run %d as %s...'%(item,run,outFile))
            dfRun.to_csv(outFile)


# %% Calculate age with fraction-of-years, calculated from date of task and DOB.
# Adapted from GetRecoveryNimhAgeWithFractions.py.
dfAll = pd.read_csv('%s/Mmi-RecoveryNimh-run1_Ratings.csv'%procDataDir)
#dfSurvey = pd.read_csv('%s/Mmi-RecoveryNimh_Survey.csv'%procDataDir)

if ('DOB' in dfSurvey) and (dfSurvey['DOB'].dtype=='object'): # try to calculate age from DOB
    print('==== Adjusting survey age to be in fraction of years at date of first task =====')
    dfSurvey['age'] = np.nan
    for iLine in range(dfSurvey.shape[0]):
        isThis = dfAll['participant']==dfSurvey.loc[iLine,'participant'] # find this participant in ratings table
        if np.any(isThis):
            ageInYears = AgeFromDateStrings(dfAll.loc[isThis,'date'].values[0], dfSurvey.loc[iLine,'DOB']) # calculate age
        dfSurvey.loc[iLine,'age'] = ageInYears # add to survey table
else:
    print('DOB not present in demographics file - using reported age instead.')

# Make sure all ages have been calculated
assert not np.any(np.isnan(dfSurvey['age'].values))

# Save file for all runs together
outFile = '%s/Mmi-RecoveryNimh_Survey.csv'%procDataDir
print('==== Saving Survey results as %s...'%(outFile))
if os.path.exists(outFile) and not overwrite:
    print('Not overwriting existing file.')
else:
    dfSurvey.to_csv(outFile)
# Save files for each run separately
for iRun in range(maxNRuns):
    run = iRun+1
    outFile = '%s/Mmi-RecoveryNimh-run%d_Survey.csv'%(procDataDir,run)
    print('==== Saving Survey run %d as %s...'%(run,outFile))
    if os.path.exists(outFile) and not overwrite:
        print('Not overwriting existing file.')
    else:
        dfSurvey.to_csv(outFile)
        print('Done!')
