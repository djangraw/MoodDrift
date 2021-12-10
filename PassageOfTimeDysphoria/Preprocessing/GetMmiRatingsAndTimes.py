#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GetMmiRatingsAndTimes.py

Created on Tue Apr 21 09:05:54 2020

@author: jangrawdc
- Updated 6/2/20 by DJ to fill in missing responses in Numbers version
- Updated 4/2/21 by DJ - changed Numbers doRatings excel path for shared code structure.
- Updated 10/29/21 by DJ - added dfProbe output for MW, boredom, and activities sliders.
- Updated 12/10/21 by DJ - added block column to dfProbe
"""

# Import packages
import pandas as pd
import numpy as np
import os

# Declare main function
def GetMmiRatingsAndTimes(inFile):

    # Read input
    print('Reading data from %s...'%inFile)
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
            dorating_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'doRatingConditions.xlsx')
            dfDoRating = pd.read_excel(dorating_path)
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
    # set up dfProbe (answers to MW, boredom, or activity probe questions)
    iProbeRating = -1
    isMWPresent = ('MWSlider_before.response' in dfIn.columns)
    isBoredomPresent = ('rateBoredomSlider.response' in dfIn.columns)
    isActivityPresent = ('activitiesSlider.response' in dfIn.columns)
    probeCols = ['participant','iBlock','iProbe','time','question','rating','RT']
    if isMWPresent:
        nProbeRatings = np.sum(pd.notna(dfIn.loc[:,'MWSlider_before.response'])) + np.sum(pd.notna(dfIn.loc[:,'MWSlider_after.response']))
    elif isBoredomPresent:
        nProbeRatings = np.sum(pd.notna(dfIn.loc[:,'rateBoredomSlider.response']))
    elif isActivityPresent:
        nProbeRatings = np.sum(pd.notna(dfIn.loc[:,'activitiesSlider.response']))
    else:
        nProbeRatings = 0
    dfProbe = pd.DataFrame(np.zeros((nProbeRatings,len(probeCols))),columns=probeCols)
    dfProbe['participant'] = participant

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

        # Add MW responses
        if isMWPresent:
            if pd.notna(dfIn.loc[iLine,'MWSlider_before.response']):
                # increment counters
                iProbeRating = iProbeRating + 1
                # Update times
                RT = dfIn.loc[iLine,'MWResp_before.rt']
                tNow = tNow + RT
                # Add info to table
                dfProbe.loc[iProbeRating,'iBlock'] = iBlock
                dfProbe.loc[iProbeRating,'iProbe'] = iProbeRating
                dfProbe.loc[iProbeRating,'rating'] = dfIn.loc[iLine,'MWSlider_before.response']
                dfProbe.loc[iProbeRating,'RT'] = RT
                dfProbe.loc[iProbeRating,'time'] = tNow
                dfProbe.loc[iProbeRating,'question'] = dfIn.loc[iLine,'questionMW']
            elif pd.notna(dfIn.loc[iLine,'MWSlider_after.response']):
                # increment counters
                iProbeRating = iProbeRating + 1
                # Update times
                RT = dfIn.loc[iLine,'MWResp_after.rt']
                tNow = tNow + RT
                # Add info to table
                dfProbe.loc[iProbeRating,'iBlock'] = iBlock
                dfProbe.loc[iProbeRating,'iProbe'] = iProbeRating
                dfProbe.loc[iProbeRating,'rating'] = dfIn.loc[iLine,'MWSlider_after.response']
                dfProbe.loc[iProbeRating,'RT'] = RT
                dfProbe.loc[iProbeRating,'time'] = tNow
                dfProbe.loc[iProbeRating,'question'] = dfIn.loc[iLine,'questionMW']
        elif isBoredomPresent and pd.notna(dfIn.loc[iLine,'rateBoredomSlider.response']):
            # increment counters
            iProbeRating = iProbeRating + 1
            # Update times
            RT = dfIn.loc[iLine,'rateBoredomSlider.rt']
            tNow = tNow + RT
            # Add info to table
            dfProbe.loc[iProbeRating,'iBlock'] = iBlock
            dfProbe.loc[iProbeRating,'iProbe'] = iProbeRating
            dfProbe.loc[iProbeRating,'rating'] = dfIn.loc[iLine,'rateBoredomSlider.response']
            dfProbe.loc[iProbeRating,'RT'] = RT
            dfProbe.loc[iProbeRating,'time'] = tNow
            dfProbe.loc[iProbeRating,'question'] = dfIn.loc[iLine,'statement']
        elif isActivityPresent and pd.notna(dfIn.loc[iLine,'activities_resp.rt']):
            # Update times
            actDur = dfIn.loc[0,'actDur'] # typically 420 s
            RT = dfIn.loc[iLine,'activities_resp.rt']
            print(f'Adding {actDur}s activities block and {RT}s RT to current time...')
            tNow = tNow + actDur + RT
            
        elif isActivityPresent and pd.notna(dfIn.loc[iLine,'activitiesSlider.response']):
            # increment counters
            iProbeRating = iProbeRating + 1
            # Update times
            RT = dfIn.loc[iLine,'activitiesSlider.rt']
            tNow = tNow + RT
            # Add info to table
            dfProbe.loc[iProbeRating,'iBlock'] = iBlock
            dfProbe.loc[iProbeRating,'iProbe'] = iProbeRating
            dfProbe.loc[iProbeRating,'rating'] = dfIn.loc[iLine,'activitiesSlider.response']
            dfProbe.loc[iProbeRating,'RT'] = RT
            dfProbe.loc[iProbeRating,'time'] = tNow
            dfProbe.loc[iProbeRating,'question'] = dfIn.loc[iLine,'questionAct']

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
            # If target happiness is not present, make it defautl to nans (mimicking old Pandas functionality)
            if 'targetHappiness' not in dfIn.columns:
                dfIn['targetHappiness'] = np.nan
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


    # Return results
    return dfTrial,dfRating,dfLifeHappy,dfProbe
