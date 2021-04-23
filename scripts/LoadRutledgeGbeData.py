#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
LoadRutledgeGbeData.py
Code to load "mobile app" data from Robb Rutledge's Great Brain Experiment

Created on Tue Jun 23 12:17:41 2020
@author: jangrawdc
-Updated 7/31/20 by DJ - added pctNoMoves column
-Updated 1/22/20 by DJ - added secondTimeSubmitted column to summary
-Updated 4/8/21 by DJ - adapted for shared code structure.
-Updated 4/22/21 by DJ - changed to use new dataset published publicly by Rutledge lab
"""
# %% Extract data points of interest from .mat file

# Import packages
import scipy.io
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Get in & out directories
dataDir = '../Data' # directory where data is found
outDir = '%s/OutFiles'%dataDir # directory where processed data files go
outFigDir = '../Figures' # directory where figures should go

# %% Load new data
        
inFile = '%s/PilotData/Rutledge_GBE_risk_data.mat'%dataDir # Path to mobile app data file

# Load full dataset
print('Loading full dataset from %s...'%inFile)
allData = scipy.io.loadmat(inFile, mdict=None, squeeze_me=True)['subjData']
# crop dataset to make it match the original version used in our paper
inFile = '../Data/PilotData/Rutledge_GBE_IDs.csv'
print('Loading IDs of included subjects from %s...'%inFile)
subjIDs = pd.read_csv(inFile,header=None).values[:,0]
print('Cropping...')
allData_new = allData[subjIDs-1] # IDs from Rutledge lab are 1-based, so we subtract 1 for python's zero-based numbering
# New dataset distributed by Rutledge lab has 14 fields:
#    ID, age, isFemale, location, lifeSatisfaction, education, nativeLanguage, deviceType, nPlays, timesPlayed, dayNumber, designVersion, datHeader, data
# The original dataset used in this paper had 6 fields:
#    Location, timeSubmitted, appversion, data, noPlays, life satisfaction
old_field_indices = [3,0,11,13,8,4] # time submitted not present in new dataset... Subbing ID instead.
allData = [[thisData[index] for index in old_field_indices] for thisData in allData_new] 

print('Done!')

# %% Extract fields

# get one-time data
noPlays = np.array([thisData[4] for thisData in allData])
lifeSatisfaction = np.array([thisData[5] for thisData in allData])/10.0 # rescale to 0-1

# get each-play data from FIRST run
location = np.array([thisData[0] for thisData in allData])
location[noPlays>1] = [loc for loc in location[noPlays>1]]
allTimesSubmitted = np.array([thisData[1] for thisData in allData])
timeSubmitted = allTimesSubmitted.copy()
timeSubmitted[noPlays>1] = [loc for loc in timeSubmitted[noPlays>1]]
secondTimeSubmitted = np.array(['']*len(timeSubmitted),dtype='object')
secondTimeSubmitted[noPlays>1] = [loc for loc in allTimesSubmitted[noPlays>1]]
appVersion = np.array([thisData[2] for thisData in allData])
appVersion[noPlays>1] = [loc for loc in appVersion[noPlays>1]]

# Get happyData from FIRST run
happyData_list = np.array([thisData[3] for thisData in allData])
nSubj = happyData_list.size
nTrials = happyData_list[0].shape[0]
nCols = happyData_list[0].shape[1]

happyData = np.zeros((nSubj,nTrials,nCols))*np.nan
for i in range(nSubj):
    if noPlays[i]==1:
        happyData[i,:,:] = happyData_list[i]
    else:
        happyData[i,:,:] = happyData_list[i][0]


#%% Extract trial-wise rating info

iRating = np.where(~np.isnan(happyData[0,:,9]))[0]
nRatings = len(iRating)
allRatings = np.zeros((nRatings,nSubj))
allWinnings = np.zeros((nTrials,nSubj))
# amounts
allWins = np.zeros((nTrials,nSubj))
allLosses = np.zeros((nTrials,nSubj))
allCertains = np.zeros((nTrials,nSubj))

allRPEs = np.zeros((nTrials,nSubj))
allGamble = np.zeros((nTrials,nSubj),dtype=int)
allTimes = np.zeros((nTrials,nSubj))
allRTs = np.zeros((nTrials,nSubj))
allRatingMoves = np.zeros((nRatings,nSubj))
allRatingRTs = np.zeros((nRatings,nSubj))
staticTrialDur = 2 # rough estimate from viewing game
for iSubj in range(nSubj):

    allCertains[:,iSubj] = happyData[iSubj,:,2]
    allWins[:,iSubj] = happyData[iSubj,:,3]
    allLosses[:,iSubj] = happyData[iSubj,:,4]

    isWin = (happyData[iSubj,:,7]==happyData[iSubj,:,3])
    isLoss = (happyData[iSubj,:,7]==happyData[iSubj,:,4])
    allGamble[:,iSubj] = happyData[iSubj,:,6]
    allWinnings[:,iSubj] = happyData[iSubj,:,7]
    allRPEs[isWin,iSubj] = happyData[iSubj,isWin,3] - happyData[iSubj,isWin,4]
    allRPEs[isLoss,iSubj] = happyData[iSubj,isLoss,4] - happyData[iSubj,isLoss,3]
    allRTs[:,iSubj] = happyData[iSubj,:,8]
    allTimes[:,iSubj] = np.cumsum(np.nan_to_num(np.abs(happyData[iSubj,:,13])) + 
                                    np.nan_to_num(happyData[iSubj,:,11]) + 
                                    np.nan_to_num(happyData[iSubj,:,8]) + 
                                    staticTrialDur)
    allRatings[:,iSubj] = happyData[iSubj,iRating,9]
    allRatingMoves[:,iSubj] = happyData[iSubj,iRating,9] - happyData[iSubj,iRating,10];
    allRatingRTs[:,iSubj] = happyData[iSubj,iRating,11]

allRatingTimes = allTimes[iRating,:]
allRatingTimes[0,:] = 0;

print('Adjusting iRating to be trial before which rating was presented')
iRating[1:] = iRating[1:]+1
print('Done!')
pctNoMoves = np.mean(allRatingMoves==0,axis=0)*100

# %% Save subject info
participants = np.arange(nSubj)+50000 # first subject = 50000 
columns = ['participant','location','timeSubmitted','appVersion','lifeHappy','noPlays','secondTimeSubmitted']
dfSummary = pd.DataFrame(np.zeros((nSubj,len(columns))),columns=columns)
dfSummary['participant'] = participants
dfSummary['location'] = location
dfSummary['timeSubmitted'] = timeSubmitted
dfSummary['secondTimeSubmitted'] = secondTimeSubmitted
dfSummary['appVersion'] = appVersion
dfSummary['lifeHappy'] = lifeSatisfaction
dfSummary['noPlays'] = noPlays
dfSummary['pctNoMoves'] = pctNoMoves # % ratings where slider wasn't moved

# save results
outFile = '%s/Mmi-GBE_Summary.csv'%(outDir)
print('Saving subject info as %s...'%outFile)
dfSummary.to_csv(outFile)
print('Done!')

# %% Place ratings & trial results in pandas dataframes
columns = ['participant','iBlock','iTrial','time','rating','RT']
dfRatings = pd.DataFrame(np.zeros((nSubj*nRatings,len(columns))),columns=columns)
dfRatings['participant'] = dfRatings['participant'].astype(int)
dfRatings['iBlock'] = dfRatings['iBlock'].astype(int)
dfRatings['iTrial'] = dfRatings['iTrial'].astype(int)

columns = ['participant', 'iBlock', 'iTrial', 'trialType',
       'lastHappyRating', 'targetHappiness', 'time', 'choice', 'RT',
       'RPE', 'outcome', 'winAmount','loseAmount','certainAmount',
       'outcomeAmount', 'currentWinnings','isRatingTrial', 'rating', 
       'ratingRT', 'ratingTime']
dfTrial = pd.DataFrame(np.zeros((nSubj*nTrials,len(columns))),columns=columns)
dfTrial['participant'] = dfTrial['participant'].astype(int)
dfTrial['iTrial'] = dfTrial['iTrial'].astype(int)
dfTrial['isRatingTrial'] = dfTrial['isRatingTrial'].astype(bool)
dfTrial['iBlock'] = dfTrial['iBlock'].astype(int)
#dfTrial['choice'] = ''
dfTrial['trialType'] = 'random'
dfTrial[['lastHappyRating', 'targetHappiness', 'time', 'choice', 'RT',
       'RPE', 'outcome', 'outcomeAmount', 'currentWinnings',
       'isRatingTrial', 'rating', 'ratingRT', 'ratingTime']] = np.nan

isRating = ~np.isnan(happyData[0,:,9])
choices = np.array(['certain','gamble'])
iTrials = np.arange(nTrials)


dfRatings['participant'] = np.repeat(participants,nRatings)
dfRatings['iBlock'] = 0
dfRatings['iTrial'] = np.tile(iRating-1, nSubj)
dfRatings['time'] = allRatingTimes.flatten('F')
dfRatings['rating'] = allRatings.flatten('F')
dfRatings['RT'] = allRatingRTs.flatten('F')

dfTrial['participant'] = np.repeat(participants,nTrials)
dfTrial['iTrial'] = np.tile(iTrials,nSubj)
dfTrial['time'] = allTimes.flatten('F')
dfTrial['choice'] = allGamble.flatten('F')
dfTrial['RT'] = allRTs.flatten('F')
dfTrial['winAmount'] = allWins.flatten('F')
dfTrial['loseAmount'] = allLosses.flatten('F')
dfTrial['certainAmount'] = allCertains.flatten('F')
dfTrial['RPE'] = allRPEs.flatten('F')
dfTrial['outcomeAmount'] = allWinnings.flatten('F')
dfTrial['currentWinnings'] = np.cumsum(allWinnings,axis=0).flatten('F')
dfTrial['isRatingTrial'] = np.tile(isRating,nSubj)

dfTrial['outcome'] = 'certain'
dfTrial.loc[dfTrial.outcomeAmount==dfTrial.winAmount,'outcome'] = 'win'
dfTrial.loc[dfTrial.outcomeAmount==dfTrial.loseAmount,'outcome'] = 'lose'


dfTrial.loc[dfTrial.isRatingTrial,'rating'] = allRatings.flatten('F')
dfTrial.loc[dfTrial.isRatingTrial,'ratingRT'] = allRatingRTs.flatten('F')
dfTrial.loc[dfTrial.isRatingTrial,'ratingTime'] = allRatingTimes.flatten('F')    

print('Converting choice column from int to string...')
dfTrial.loc[dfTrial.choice==0,'choice'] = 'certain'
dfTrial.loc[dfTrial.choice==1,'choice'] = 'gamble'

print('Done!')

# %% Select 5000 participants as exploratory sample

# Randomly (but repeatably) select 5k subjects
nExp = 5000 # number of exploratory participants
randSeed = 24567 # seed used to get a repeatable set of participants
np.random.seed(randSeed)
# use these 5k subjects as the exploratory mobile app cohort
exploratoryParticipants = np.random.choice(participants,nExp,replace=False)
dfRatings_exp = dfRatings.loc[np.isin(dfRatings.participant,exploratoryParticipants),:]
dfTrial_exp = dfTrial.loc[np.isin(dfTrial.participant,exploratoryParticipants),:]
dfSummary_exp = dfSummary.loc[np.isin(dfSummary.participant,exploratoryParticipants),:]
# Use the rest as confirmatory sample
dfRatings_conf = dfRatings.loc[~np.isin(dfRatings.participant,exploratoryParticipants),:]
dfTrial_conf = dfTrial.loc[~np.isin(dfTrial.participant,exploratoryParticipants),:]
dfSummary_conf = dfSummary.loc[~np.isin(dfSummary.participant,exploratoryParticipants),:]

# %% Save results as csv files

# Save exploratory mobile app cohort
outFile = '%s/Mmi-GbeExplore_Ratings.csv'%(outDir)
print('Saving ratings as %s...'%outFile)
dfRatings_exp.to_csv(outFile)
print('Done!')

outFile = '%s/Mmi-GbeExplore_Trial.csv'%(outDir)
print('Saving trials as %s...'%outFile)
dfTrial_exp.to_csv(outFile)
print('Done!')

outFile = '%s/Mmi-GbeExplore_Summary.csv'%(outDir)
print('Saving subject info as %s...'%outFile)
dfSummary_exp.to_csv(outFile)
print('Done!')

# Same for confirmatory mobile app cohort
outFile = '%s/Mmi-GbeConfirm_Ratings.csv'%(outDir)
print('Saving ratings as %s...'%outFile)
dfRatings_conf.to_csv(outFile)
print('Done!')

outFile = '%s/Mmi-GbeConfirm_Trial.csv'%(outDir)
print('Saving trials as %s...'%outFile)
dfTrial_conf.to_csv(outFile)
print('Done!')

outFile = '%s/Mmi-GbeConfirm_Summary.csv'%(outDir)
print('Saving subject info as %s...'%outFile)
dfSummary_conf.to_csv(outFile)
print('Done!')

# %% Find runs where mood ratings were ignored (could be used to exclude later)

#isFlat = np.all(np.diff(allRatings,axis=0)==0, axis=0)
isFlat = np.all(allRatingMoves==0, axis=0)
print('%d flat subjects.'%np.sum(isFlat))

# %% Plot ratings

# collect summary stats
meanRatings = np.nanmean(allRatings,axis=1)
steRatings = np.nanstd(allRatings,axis=1)/np.sqrt(nSubj)
medianRatings = np.nanmedian(allRatings,axis=1)
medianRatingTimes = np.nanmedian(allRatingTimes,axis=1)

plt.figure(623,figsize=[10,10],dpi=180); plt.clf();
#Plot mean ratings
plt.subplot(2,2,1)
plt.plot(medianRatingTimes, meanRatings,'.-', label='mean+/- ste',zorder=5)
plt.fill_between(medianRatingTimes,meanRatings-steRatings,meanRatings+steRatings,
                 alpha=0.5,zorder=0)
plt.plot(medianRatingTimes,medianRatings,'.-',label='median')

# Annotate plot
plt.xlabel('mean trial time')
plt.ylabel('mean happiness rating')
plt.title('Rutledge Data (n=%d)'%nSubj)
plt.legend()

# Examine skew of 1st, 2nd, & last ratings
plt.subplot(2,2,2)
xHist = np.linspace(-0.025,1.025,22)
plt.hist(allRatings[0,:],xHist,alpha=0.5,label='first')
plt.hist(allRatings[1,:],xHist,alpha=0.5,label='second')
plt.hist(allRatings[-1,:],xHist,alpha=0.5,label='last')
plt.xlabel('happiness rating')
plt.ylabel('# subjects')
plt.title('Individual happiness ratings (n=%d)'%nSubj)
plt.legend()

# Make 2D histogram
plt.subplot(2,2,3)
# Create heatmap
x = np.tile(np.arange(nRatings),(nSubj,1)).T.flatten();
y = allRatings.flatten();
xedges = np.arange(nRatings+1)-0.5
yedges = np.linspace(-0.025,1.025,22)
heatmap, xedges, yedges = np.histogram2d(x, y, bins=(xedges,yedges))
heatmap = heatmap/nSubj*100;
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
# plot
plt.imshow(np.rot90(heatmap), extent=extent, cmap='jet',aspect='auto') # plot heatmap (must be rotated 90deg for some reason!)
# annotate plot    
plt.colorbar().set_label('% subjects');
plt.axhline(y=0.5,linestyle=':',color='w')
plt.title('2D histogram of ratings (n=%d)'%(nSubj))
plt.xlabel('rating number')
plt.ylabel('happiness rating');

# Make spaghetti plot of a few subjects
plt.subplot(2,2,4)
nToPlot = 30
plt.plot(iRating, allRatings[:,:nToPlot],'.-')
plt.xlabel('trial number')
plt.ylabel('happiness rating')
plt.title('Individual ratings of first %d subjects'%nToPlot)
plt.tight_layout()

# Save result
outFile = '%s/RutledgeGbeRatings.png'%outFigDir
print('Saving figure as %s...'%outFile)
plt.savefig(outFile)
print('Done!')


# %% Plot winnings

# collect summary stats
meanWinnings = np.nanmean(allWinnings,axis=1)
steWinnings = np.nanstd(allWinnings,axis=1)/np.sqrt(nSubj)
medianWinnings = np.nanmedian(allWinnings,axis=1)

plt.figure(624); plt.clf();
#Plot winnings
plt.plot(range(nTrials),meanWinnings,'.-', label='mean+/- ste',zorder=5)
plt.fill_between(range(nTrials),meanWinnings-steWinnings,meanWinnings+steWinnings,
                 alpha=0.5,zorder=0)
plt.plot(range(nTrials),medianWinnings,'.-',label='median')

# Annotate plot
plt.xlabel('trial number')
plt.ylabel('mean winnings')
plt.title('Rutledge Data (n=%d)'%nSubj)
plt.legend()
plt.tight_layout()

# Save result
outFile = '%s/RutledgeGbeWinnings.png'%outFigDir
print('Saving figure as %s...'%outFile)
plt.savefig(outFile)
print('Done!')

