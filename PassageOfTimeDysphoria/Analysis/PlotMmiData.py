#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 09:38:37 2020

@author: jangrawdc

- Updated 10/29/20 by DJ - made axis labels have only first letter capitalized
- Updated 3/31/21 by DJ - adapted for shared code structure.
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os

def GetBlockTimes(dfTrial,dfRating):
    '''
    Get the times at which block switches occurred.
    '''
    # Get block times
    iBlocks = np.unique(dfRating.iBlock).astype(int)
    nBlocks = np.max(iBlocks)+1
    tBlockSwitch = np.ones(nBlocks+1)*np.nan;
    blockType = ['']*nBlocks
    tBlockSwitch[0] = 0;
    for iBlock in iBlocks:
        iLast = np.where(dfRating.iBlock==iBlock)[0][-1]
        if iLast<dfRating.shape[0]-1:
            tBlockSwitch[iBlock+1] = dfRating.time[iLast+1]-3
        else:
            tBlockSwitch[iBlock+1] = dfRating.time[iLast]+3
        iLast = np.where(dfTrial.iBlock==iBlock)[0][-1]
        if (type(dfTrial.loc[iLast,'targetHappiness'])==np.float64) and not np.isnan(dfTrial.loc[iLast,'targetHappiness']):
            blockType[iBlock] = '%s (target=%g)'%(dfTrial.trialType[iLast],dfTrial.targetHappiness[iLast])
        else:
            blockType[iBlock] = dfTrial.trialType[iLast]

    return tBlockSwitch,blockType


def PlotBlockTimes(tBlockSwitch,blockType,yLim=[0,1]):
    '''
    Plot the times at which blocks started and ended, with text labels.
    '''
    try:
        yMin,yMax=yLim
    except ValueError:
        print('Inferring y limits from plot.')
        yMin,yMax = plt.gca().get_ylim()

    for iBlock in range(len(blockType)):
        blockText = '\n'.join(blockType[iBlock].split(' '))
        plt.text((tBlockSwitch[iBlock]+tBlockSwitch[iBlock+1])/2 ,yMin+(yMax-yMin)*0.99,
                 blockText,ha='center',va='top');
    plt.ylim(yMin,yMax)
    plt.vlines(tBlockSwitch,yMin,yMax,linestyles=':')


def PlotMmiRatings(dfTrial,dfRating,interp='step',autoYlim=False,doBlockLines=True,ratingLabel=''):
    '''
    Plot mood ratings for an MMI ratings struct.
    '''
    # get info
    participants = np.unique(dfRating.participant)
    if doBlockLines:
        tBlockSwitch,blockType = GetBlockTimes(dfTrial,dfRating)

    for participant in participants:
        if len(participants)>1:
            ratingLabel = participant

        isThis = (dfRating.participant==participant)
        if interp=='step':
            ratingLine = plt.step(dfRating.loc[isThis,'time'],dfRating.loc[isThis,'rating'],'-',where='post')[0]
            plt.plot(dfRating.loc[isThis,'time'],dfRating.loc[isThis,'rating'],'.',color=ratingLine.get_c(),label=ratingLabel)
        else:
            plt.plot(dfRating.loc[isThis,'time'],dfRating.loc[isThis,'rating'],'.-',label=ratingLabel)

        # add ste patches
        if 'steRating' in dfRating.columns:
            if interp=='step':
                plt.fill_between(dfRating.loc[isThis,'time'],dfRating.loc[isThis,'rating']-dfRating.loc[isThis,'steRating'],
                                 dfRating.loc[isThis,'rating']+dfRating.loc[isThis,'steRating'],
                                 alpha=0.5,zorder=0,step='post')
            else:
                plt.fill_between(dfRating.loc[isThis,'time'],dfRating.loc[isThis,'rating']-dfRating.loc[isThis,'steRating'],
                                 dfRating.loc[isThis,'rating']+dfRating.loc[isThis,'steRating'],
                                 alpha=0.5,zorder=0)

    # add block lines & text
    if autoYlim:
        if doBlockLines:
            PlotBlockTimes(tBlockSwitch,blockType,yLim=plt.gca().get_ylim())
    else:
        if doBlockLines:
            PlotBlockTimes(tBlockSwitch,blockType,yLim=[0,1])
        plt.ylim(0,1)

    # annotate plot
    plt.xlabel('Time (s)')
    plt.ylabel('Mood rating (0-1)')
    if len(participants)==1:
        plt.title('MMI participant %s'%participants[0])
    else:
        plt.legend()
        plt.title('MMI, %d participants'%len(participants))


def PlotMmiRPEs(dfTrial,dfRating,doBlockLines=True):
    '''
    Plot RPEs for an MMI trial struct.
    '''
    # get info
    participants = np.unique(dfTrial.participant)
    tBlockSwitch,blockType = GetBlockTimes(dfTrial,dfRating)

    # plot
    for participant in participants:
        isThis = (dfTrial.participant==participant)
        plt.plot(dfTrial.loc[isThis,'time'],dfTrial.loc[isThis,'RPE'],'.-',label=participant)


        # add win/loss/certain indicators
        plt.plot(dfTrial.loc[isThis & (dfTrial.outcome=='win'),'time'],dfTrial.loc[isThis & (dfTrial.outcome=='win'),'RPE'],'go')
        plt.plot(dfTrial.loc[isThis & (dfTrial.outcome=='lose'),'time'],dfTrial.loc[isThis & (dfTrial.outcome=='lose'),'RPE'],'rs')
        plt.plot(dfTrial.loc[isThis & (dfTrial.outcome=='certain'),'time'],dfTrial.loc[isThis & (dfTrial.outcome=='certain'),'RPE'],'k+')

        # a add ste patches
        if 'steRPE' in dfTrial.columns:
            plt.fill_between(dfTrial.loc[isThis,'time'],dfTrial.loc[isThis,'RPE']-dfTrial.loc[isThis,'steRPE'],
                             dfTrial.loc[isThis,'RPE']+dfTrial.loc[isThis,'steRPE'],alpha=0.5,zorder=0)

    # add zero line
    plt.axhline(0,c='k',ls='--',zorder=-1)

    # add block lines & text
    if doBlockLines:
        PlotBlockTimes(tBlockSwitch,blockType,yLim=plt.gca().get_ylim())

    # annotate plot
    plt.xlabel('Time (s)')
    plt.ylabel('RPE')
    if len(participants)==1:
        plt.title('MMI participant %s'%participants[0])
    else:
        plt.legend()
        plt.title('MMI, %d participants'%len(participants))



def PlotMmiOutcomes(dfTrial,dfRating,doBlockLines=True):
    '''
    Plot trial-wise outcome amounts for an MMI trial struct.
    '''
    # get info
    participants = np.unique(dfTrial.participant)
    tBlockSwitch,blockType = GetBlockTimes(dfTrial,dfRating)

    # plot
    for participant in participants:
        isThis = (dfTrial.participant==participant)
        plt.plot(dfTrial.loc[isThis,'time'],dfTrial.loc[isThis,'outcomeAmount'],'.-',label=participant)

        # add ste patches
        if 'steOutcomeAmount' in dfTrial.columns:
            plt.fill_between(dfTrial.loc[isThis,'time'],dfTrial.loc[isThis,'outcomeAmount']-dfTrial.loc[isThis,'steOutcomeAmount'],
                             dfTrial.loc[isThis,'outcomeAmount']+dfTrial.loc[isThis,'steOutcomeAmount'],alpha=0.5,zorder=0)

    # add zero line
    plt.axhline(0,c='k',ls='--',zorder=-1)

    # add block lines & text
    if doBlockLines:
        PlotBlockTimes(tBlockSwitch,blockType,yLim=plt.gca().get_ylim())

    # annotate plot
    plt.xlabel('Time (s)')
    plt.ylabel('Outcome amount')
    if len(participants)==1:
        plt.title('MMI participant %s'%participants[0])
    else:
        plt.legend()
        plt.title('MMI, %d participants'%len(participants))


def PlotMmiWinnings(dfTrial,dfRating,doBlockLines=True):
    '''
    Plot winnings for an MMI trial struct.
    '''
    # get info
    participants = np.unique(dfTrial.participant)
    tBlockSwitch,blockType = GetBlockTimes(dfTrial,dfRating)

    # plot
    for participant in participants:
        isThis = (dfTrial.participant==participant)
        plt.plot(dfTrial.loc[isThis,'time'],dfTrial.loc[isThis,'currentWinnings'],'.-',label=participant)

        # add ste patches
        if 'steCurrentWinnings' in dfTrial.columns:
            plt.fill_between(dfTrial.loc[isThis,'time'],dfTrial.loc[isThis,'currentWinnings']-dfTrial.loc[isThis,'steCurrentWinnings'],
                             dfTrial.loc[isThis,'currentWinnings']+dfTrial.loc[isThis,'steCurrentWinnings'],alpha=0.5,zorder=0)

    # add zero line
    plt.axhline(0,c='k',ls='--',zorder=-1)

    # add block lines & text
    if doBlockLines:
        PlotBlockTimes(tBlockSwitch,blockType,yLim=plt.gca().get_ylim())

    # annotate plot
    plt.xlabel('Time (s)')
    plt.ylabel('Current winnings')
    if len(participants)==1:
        plt.title('MMI participant %s'%participants[0])
    else:
        plt.legend()
        plt.title('MMI, %d participants'%len(participants))



def GetMeanRatings(dfRating,nRatings=-1,participantLabel='mean',doInterpolation=False):
    '''
    Get rating-wise means of mood ratings and the times at which they
    were recieved.
    '''
    # Get constants
    participants = np.unique(dfRating.participant)
    nSubj = participants.size
    # detect # ratings automatically
    if nRatings<0:
        # see if everyone has the same number of ratings
        nRatings_subj = np.zeros(nSubj,int)
        lastRatingTime = np.zeros(nSubj)
        firstRatingTime = np.zeros(nSubj)
        for i,participant in enumerate(participants):
            nRatings_subj[i] = np.sum(dfRating.participant==participant)
            firstRatingTime[i] = dfRating.loc[dfRating.participant==participant,'time'].values[0]
            lastRatingTime[i] = dfRating.loc[dfRating.participant==participant,'time'].values[-1]
        # if so, use that number as nRatings
        if np.all(nRatings_subj==nRatings_subj[0]) and not doInterpolation:
            nRatings = nRatings_subj[0]
#            doInterpolation = False; # do interpolation only if the input says so
        else: # otherwise, use nRatings=-1 to flag interpolation
            doInterpolation = True; # force interpolation because the # of ratings is not consistent
            tFirst = int(np.max(firstRatingTime))
            nRatings = int(np.min(lastRatingTime)) - tFirst
    elif doInterpolation: # used to be if...
        lastRatingTime = np.zeros(nSubj)
        firstRatingTime = np.zeros(nSubj)
        for i,participant in enumerate(participants):
            firstRatingTime[i] = dfRating.loc[dfRating.participant==participant,'time'].values[0]
            lastRatingTime[i] = dfRating.loc[dfRating.participant==participant,'time'].values[nRatings-1]
        tFirst = int(np.max(firstRatingTime))
        nRatings = int(np.min(lastRatingTime)) - tFirst

    # Set up
    allRating = np.zeros((nRatings,nSubj))
    allTimes = np.zeros((nRatings,nSubj))
    # Collect values in array
    for i,participant in enumerate(participants):
        if doInterpolation:
            theseRatings = dfRating.loc[dfRating.participant==participant, 'rating'].values
            theseTimes = dfRating.loc[dfRating.participant==participant, 'time'].values
            allTimes[:,i] = np.arange(tFirst,tFirst+nRatings)
            allRating[:,i] = np.interp(allTimes[:,i],theseTimes,theseRatings)
        else:
            allRating[:,i] = dfRating.loc[dfRating.participant==participant, 'rating'].values[:nRatings]
            allTimes[:,i] = dfRating.loc[dfRating.participant==participant, 'time'].values[:nRatings]

    # Compile into dataframe
    cols = dfRating.columns
    dfRatingMean = pd.DataFrame(np.ones((nRatings,len(cols)))*np.nan,columns=cols)
#    dfRatingMean = dfRating.loc[dfRating.participant==participants[0],:].copy()
#    dfRatingMean = dfRatingMean.iloc[:nRatings,:]
    dfRatingMean['participant'] = '%s (n=%s)'%(participantLabel,nSubj)
    dfRatingMean['iBlock'] = dfRating['iBlock'].values[0]
    dfRatingMean['iTrial'] = np.arange(nRatings)
    dfRatingMean['rating'] = np.mean(allRating,axis=1)
    dfRatingMean['time'] = np.mean(allTimes,axis=1)
    dfRatingMean['steRating'] = np.std(allRating,axis=1)/np.sqrt(nSubj)
    dfRatingMean['steTime'] = np.std(allTimes,axis=1)/np.sqrt(nSubj)
    # Return result
    dfRatingMean = dfRatingMean.reset_index(drop=True)
    return dfRatingMean

def GetMeanTrials(dfTrial,nTrials=-1,participantLabel='mean'):
    '''
    Get trial-wise mean of RPEs and the times at which they were received.
    '''
    # Import binomial confidence interval function
    from statsmodels.stats.proportion import proportion_confint

    # Get constants
    participants = np.unique(dfTrial.participant)
    nSubj = participants.size
    if nTrials<0:
        # see if everyone has the same number of ratings
        nTrials_subj = np.zeros(nSubj,int)
        for i,participant in enumerate(participants):
            nTrials_subj[i] = np.sum(dfTrial.participant==participant)
        # use min trial number as nRatings
        nTrials = int(np.min(nTrials_subj))
    # Set up
    allRpes = np.zeros((nTrials,nSubj))
    allTrialTimes = np.zeros((nTrials,nSubj))
    allAmounts = np.zeros((nTrials,nSubj))
    allWinnings = np.zeros((nTrials,nSubj))
    allGamble = np.zeros((nTrials,nSubj))
    # Collect values in array
    for i,participant in enumerate(participants):
        allRpes[:,i] = dfTrial.loc[dfTrial.participant==participant, 'RPE'].values[:nTrials]
        allTrialTimes[:,i] = dfTrial.loc[dfTrial.participant==participant, 'time'].values[:nTrials]
        allAmounts[:,i] = dfTrial.loc[dfTrial.participant==participant, 'outcomeAmount'].values[:nTrials]
        allWinnings[:,i] = dfTrial.loc[dfTrial.participant==participant, 'currentWinnings'].values[:nTrials]
        allGamble[:,i] = dfTrial.loc[dfTrial.participant==participant, 'choice'].values[:nTrials]=='gamble'
    # Compile into dataframe
    dfTrialMean = dfTrial.loc[dfTrial.participant==participants[0],:].copy()
    dfTrialMean = dfTrialMean.iloc[:nTrials,:]
    dfTrialMean['RPE'] = np.mean(allRpes,axis=1)
    dfTrialMean['time'] = np.mean(allTrialTimes,axis=1)
    dfTrialMean['outcomeAmount'] = np.mean(allAmounts,axis=1)
    dfTrialMean['currentWinnings'] = np.mean(allWinnings,axis=1)
    dfTrialMean['gambleFrac'] = np.mean(allGamble,axis=1)
    gambleErrBounds = [proportion_confint(count, nSubj, alpha=0.05, method='normal') for count in np.sum(allGamble,axis=1)]
    dfTrialMean['gambleFracCIMin'] = [err[0] for err in gambleErrBounds]
    dfTrialMean['gambleFracCIMax'] = [err[1] for err in gambleErrBounds]
    dfTrialMean['participant'] = '%s (n=%s)'%(participantLabel,nSubj)
    dfTrialMean['outcome'] = ''
    dfTrialMean['steRPE'] = np.std(allRpes,axis=1)/np.sqrt(nSubj)
    dfTrialMean['steTime'] = np.std(allTrialTimes,axis=1)/np.sqrt(nSubj)
    dfTrialMean['steOutcomeAmount'] = np.std(allAmounts,axis=1)/np.sqrt(nSubj)
    dfTrialMean['steCurrentWinnings'] = np.std(allWinnings,axis=1)/np.sqrt(nSubj)

    # Return result
    dfTrialMean = dfTrialMean.reset_index(drop=True)
    return dfTrialMean

def MakeRatingGif(dfTrial,dfRating,nPerPage=10,participants='',outFigDir='../Figures',batchName='',frameRate=3,deleteTemp=True):
    '''
    Make a GIF of individual subjects' results.
    '''
    # Import gif
    from MakeGif import MakeGif
    # Sort participants in alpha order by default
    if participants=='':
        participants = np.unique(dfTrial.participant)

    # Get means
    dfRatingMean = GetMeanRatings(dfRating)
    dfTrialMean = GetMeanTrials(dfTrial)

    nSubj = participants.size
    nPages = int(np.ceil(float(nSubj)/nPerPage))
    if not os.path.exists('my_folder'):
        os.makedirs('my_folder')
    tempImageFiles = ['TEMP/GifPage%03d.png'%(iPage+1) for iPage in range(nPages)]

    print('Making TEMP directory...')
    os.mkdir('TEMP')

    print('Saving %d temporary images...'%nPages)
    for iPage in range(nPages):
        plt.clf()
        # add mean to every page
        PlotMmiRatings(dfTrialMean,dfRatingMean,interp='line')
        for iInPage in range(nPerPage):
            iSubj = iPage*nPerPage + iInPage
            if iSubj<nSubj:
                # print(participants[iSubj])
                PlotMmiRatings(dfTrial.loc[dfTrial.participant==participants[iSubj]],
                               dfRating.loc[dfRating.participant==participants[iSubj]],
                               interp='line',doBlockLines=False)
        # annotate plot
        plt.title('MMI-%s (page %03d/%03d)'%(batchName,iPage+1,nPages))
        tMax = np.max(dfRating.time)
        plt.xlim([0,tMax])
        # save temporary figure
        print('Saving %s...'%tempImageFiles[iPage])
        plt.savefig(tempImageFiles[iPage]);

    # Compile into GIF
    outFile = '%s/MmiGif-%s.gif'%(outFigDir,batchName)
    print('Saving %s...'%outFile)
    MakeGif(tempImageFiles,outFile=outFile,frameRate=frameRate,loop=0)

    # Clean up
    if deleteTemp:
        print('Deleting temporary files...')
        for iPage in range(nPages):
            os.remove(tempImageFiles[iPage])
        os.rmdir('TEMP')
    print('Done!')
