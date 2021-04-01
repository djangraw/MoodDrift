#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
GetMmiRatingsAndTimes.py

Created on Tue Apr 21 09:05:54 2020

@author: jangrawdc
Updated 6/2/20 by DJ to fill in missing responses in Numbers version
"""

import pandas as pd
import numpy as np
#from matplotlib import pyplot as plt

def GetMmiRatingsAndTimes(inFile):

    print('Reading data from %s...'%inFile)
    # %%
    dfIn = pd.read_csv(inFile);
    nLines = dfIn.shape[0];
    
    # Extract timing parameters
    timingParams = {'freqMult': 1.0,
                    'getAnswerDur':3.0,
                    'showChoiceDur':4.0,
                    'showResultDur': 1,
                    'isiDur': 1.0,
                    'moodQuestionDur':1.5, # 1.5 for motion version
                    'moodRatingDur':4.5,
                    'itiDur':2.0,
                    'nTrialsPerBlock':22,
                    'iriDur':16.5, # 16.5 for motion version
                    'nReturnReps':18,
                    'fixFreq':0.25} # for motion version
    print('Reading parameters...')
    for param in timingParams:
        if param in dfIn.columns:
            # print('   %s = %g'%(param,GetNonNanValues(dfIn,param,True)))
            timingParams[param] = dfIn.loc[pd.notna(dfIn[param]),param].values[0]
        elif param=='iriDur':
            timingParams['iriDur'] = 15.0/timingParams['freqMult'];
    print('Done! Using defaults for the rest.')
    
    # detect open/closed-loop gambling condition
    if 'condition' in dfIn.columns:
        mmiCondition = dfIn.loc[0,'condition']
        print('Detected gambling condition = %s.'%mmiCondition)
    elif 'open' in dfIn.expName[0].lower():
        mmiCondition = 'open'
        print('Inferring gambling condition = %s from experiment name.'%mmiCondition)
    else:
        mmiCondition = 'closed'
        print('Assuming gambling condition = %s.'%mmiCondition)
    
    # check if subject made no selections
    if 'choice' not in dfIn.columns:
        print('WARNING: subject never chose gamble vs. certain.')
        dfIn['choice'] = np.nan;
    # In early versions, returnLoop.ran column is missing. Infer it.
    if 'returnLoop.ran' not in dfIn.columns:
        dfIn['returnLoop.ran'] = np.nan;
        dfIn.loc[pd.notna(dfIn['happySlider.response']) & pd.isna(dfIn['choice']),'returnLoop.ran'] = 1;
        print('Inferring returnLoop.ran column... %d rest trials found.'%np.nansum(dfIn['returnLoop.ran']))
    
    # In Numbers version, sliders were replaced with keypresses. Reconstruct them.
    if 'happySlider.rt' not in dfIn.columns:
        dfIn['happySlider.rt'] = dfIn['happyResp.rt'];
        dfIn['happySlider.response'] = (dfIn['happyResp.keys']-1.0)/8.0;
        dfIn['blockHappySlider.rt'] = dfIn['blockHappyResp.rt'];
        dfIn['blockHappySlider.response'] = (dfIn['blockHappyResp.keys']-1.0)/8.0;
        dfIn['lifeHappySlider.rt'] = dfIn['lifeHappyResp.rt'];
        dfIn['lifeHappySlider.response'] = (dfIn['lifeHappyResp.keys']-1.0)/8.0;
        print('Numbers version detected. Inferring *happySlider responses from *happyResp responses.')        
        try:
            dfDoRating = pd.read_excel('/Users/jangrawdc/Documents/PRJ24_MmiOnline_Hanna-Argyris/mmi-numbers/doRatingConditions.xlsx')
            isRatingTrial = dfDoRating.doRating.values=='Yes'
            isGambleRow = pd.notna(dfIn['choice'])
            dfIn['isRatingTrial'] = False;
            dfIn.loc[pd.notna(dfIn['returnLoop.ran']),'isRatingTrial'] = True
            dfIn.loc[isGambleRow,'isRatingTrial'] = isRatingTrial[:np.sum(isGambleRow)]
            isMissingResp = (pd.isna(dfIn['happyResp.keys']) & dfIn['isRatingTrial'])
            dfIn.loc[isMissingResp,'happySlider.response'] = dfIn.loc[isMissingResp,'lastHappyRating']
            print('Filled in %d missing responses.'%np.sum(isMissingResp))
        except:
            print('WARNING: Failed to fill in missing happiness ratings.')
            
    # Calculate trial times (MINUS RESPONSE-LIMITED PERIODS)
    # rest period trials
    restDur = timingParams['iriDur']
    # gambling trials
    trialDur = timingParams['showChoiceDur'] + timingParams['showResultDur'] + timingParams['itiDur']; # showChoiceDur=4; showResultDur=1;
    # period before mood ratings during gambling trials
    trialIsiDur = timingParams['isiDur']
    
    # Read in ratings & timing line by line
    tNow = 0;
    iRating = -1;
    iTrial = -1;
    iBlock = -1;
    participant = dfIn.participant[0]
    nRatings = np.sum(pd.notna(dfIn.loc[:,'blockHappySlider.response'])) + np.sum(pd.notna(dfIn.loc[:,'happySlider.response']))
    nTrials = np.sum(pd.notna(dfIn.loc[:,'lastHappyRating']))
    cols = ['participant','iBlock','iTrial','time','rating','RT']
    dfRating = pd.DataFrame(np.zeros((nRatings,len(cols))),columns=cols)
    dfRating['participant'] = participant
    cols = ['participant','iBlock','iTrial','trialType','lastHappyRating','targetHappiness','time','choice','RT','RPE','outcome','outcomeAmount','currentWinnings','isRatingTrial','rating','ratingRT','ratingTime']
    dfTrial = pd.DataFrame(np.ones((nTrials,len(cols)))*np.nan,columns=cols)
    dfTrial['participant'] = participant
    dfTrial['isRatingTrial'] = False;
    
    # Walk down lines of input table
    for iLine in range(nLines):
        
        # Life happiness rating
        if pd.notna(dfIn.loc[iLine,'lifeHappySlider.response']):
            # record response and RT
            dfLifeHappy = pd.DataFrame({'participant': [participant],
                                        'rating': [dfIn.loc[iLine,'lifeHappySlider.response']],
                                        'RT': [dfIn.loc[iLine,'lifeHappySlider.rt']]},columns=['participant','rating','RT'])
        
        # Mood rating at start of each block
        if pd.notna(dfIn.loc[iLine,'blockHappySlider.response']):
            # increment counters
            iBlock = iBlock + 1
            iRating = iRating + 1
            # Update times
            RT = dfIn.loc[iLine,'blockHappySlider.rt']
            if pd.isna(RT):
                tNow = tNow + timingParams['moodQuestionDur'] + timingParams['moodRatingDur']
            else:
                tNow = tNow + timingParams['moodQuestionDur'] + RT
            # Add info to table
            dfRating.loc[iRating,'rating'] = dfIn.loc[iLine,'blockHappySlider.response']
            dfRating.loc[iRating,'RT'] = RT
            dfRating.loc[iRating,'time'] = tNow            
            dfRating.loc[iRating,'iBlock'] = iBlock
            dfRating.loc[iRating,'iTrial'] = iTrial            

        # Trial (with or without mood rating)
        if pd.notna(dfIn.loc[iLine,'choice']):
            iTrial = iTrial + 1
            if 'choiceKey.rt' in dfIn.columns:
                RT = dfIn.loc[iLine,'choiceKey.rt']
            else: # never responded
                RT = np.nan
            if np.isnan(RT):
                tNow = tNow + trialDur + timingParams['getAnswerDur']
            else:
                tNow = tNow + trialDur + RT
            # add to trial table
            dfTrial.loc[iTrial,'trialType'] = mmiCondition
            dfTrial.loc[iTrial,'RT'] = RT 
            dfTrial.loc[iTrial,'time'] = tNow 
            dfTrial.loc[iTrial,'iBlock'] = iBlock
            dfTrial.loc[iTrial,'iTrial'] = iTrial                
            dfTrial.loc[iTrial,['lastHappyRating','targetHappiness','choice','outcome','outcomeAmount','currentWinnings','RPE']] = dfIn.loc[iLine,['lastHappyRating','targetHappiness','choice','outcome','outcomeAmount','currentWinnings','RPE']]
            # infer target happiness from another column if it's not present
            if pd.isna(dfTrial.loc[iTrial,'targetHappiness']):
                if dfIn.loc[iLine,'isHappyBlock']:
                    dfTrial.loc[iTrial,'targetHappiness'] = 1;
                else:
                    dfTrial.loc[iTrial,'targetHappiness'] = 0;
                    
        # Trial Happiness Rating
        if pd.notna(dfIn.loc[iLine,'happySlider.response']):
            iRating = iRating + 1
            dfRating.loc[iRating,'rating'] = dfIn.loc[iLine,'happySlider.response']
            RT = dfIn.loc[iLine,'happySlider.rt']
            dfRating.loc[iRating,'RT'] = RT
            # timing: if it's a rest period, add restDur, otherwise add ISI.
            if ('returnLoop.ran' in dfIn.columns) and (pd.notna(dfIn.loc[iLine,'returnLoop.ran'])):
                # Increment counter
                iTrial = iTrial+1
                # Update time
                tNow = tNow + restDur
                # add info to table
                dfTrial.loc[iTrial,'trialType'] = 'rest'
                dfTrial.loc[iTrial,'lastHappyRating'] = dfIn.loc[iLine,'lastHappyRating']
                dfTrial.loc[iTrial,'RT'] = np.nan 
                dfTrial.loc[iTrial,'time'] = tNow 
                dfTrial.loc[iTrial,'iBlock'] = iBlock
                dfTrial.loc[iTrial,'iTrial'] = iTrial    
                dfTrial.loc[iTrial,['choice','outcome']] = ''
                dfTrial.loc[iTrial,'targetHappiness'] = np.nan
                dfTrial.loc[iTrial,'outcomeAmount'] = 0
                if iTrial==0:
                    dfTrial.loc[iTrial,'currentWinnings'] = 0
                else:
                    dfTrial.loc[iTrial,'currentWinnings'] = dfTrial.loc[iTrial-1,'currentWinnings']
            else:
                # Update time
                tNow = tNow + trialIsiDur
            # Update time
            if pd.isna(RT):
                tNow = tNow + timingParams['moodQuestionDur'] + timingParams['moodRatingDur']
            else:
                tNow = tNow + timingParams['moodQuestionDur'] + RT
            # add info to table
            dfRating.loc[iRating,'time'] = tNow
            dfRating.loc[iRating,'iBlock'] = iBlock
            dfRating.loc[iRating,'iTrial'] = iTrial
            # Add ratings to trial struct
            dfTrial.loc[iTrial,'isRatingTrial'] = True;
            dfTrial.loc[iTrial,'rating'] = dfRating.loc[iRating,'rating']
            dfTrial.loc[iTrial,'ratingRT'] = dfRating.loc[iRating,'RT']
            dfTrial.loc[iTrial,'ratingTime'] = dfRating.loc[iRating,'time']
    
        

    return dfTrial,dfRating,dfLifeHappy