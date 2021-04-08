#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 13:49:22 2020

@author: jangrawdc
"""
import pandas as pd
import numpy as np
import os.path


def GetNonNanValues(df,field,justOne=True):
    values = df.loc[pd.notna(df[field]),field].values
    if justOne:
        # check that there's just one non-nan value
        assert len(values)==1, "%d non-nan values found in field %s"%(len(values),field)
        values = values[0]
    # return result
    return values

def ImportMmiSurveyData(inFile, mTurkID=np.nan, demoDataFile='OutFiles/COVID01_DataCheck.csv'):
    # print('  Importing Survey File %s '%inFile)

    if not os.path.exists(str(inFile)):
        # dfDataCheck.loc[sdan,'isSurveyPresent'] = False
        print('*** Survey file not found!')
        return -1


    # read and extract info
    dfIn = pd.read_csv(inFile);
    # If there's no 'Page' input, add it
    if 'Page' not in dfIn.columns:
        dfIn['Page'] = np.arange(dfIn.shape[0])+1

    # hashCode = GetNonNanValues(dfIn,'hashCode',True)
    # participant = dfIn.participant[0]
    nPages = np.sum(pd.notna(dfIn.Responses)) # pages of non-COVID responses
    nDemo = 4 # number of demographics q's
    if 'covidPage' in dfIn.columns:
        nCovid = 27; # number of covid questions
    else:
        nCovid = 0;

    # Set up
    nQ = 6*nPages+nDemo+nCovid
    dfQandA = pd.DataFrame(np.zeros((nQ,8)),columns=['Survey','SurveyQNum','Page','Question','iAnswer','Answer','RT','iCatchAnswer'])
    for intCol in ['SurveyQNum','Page','iAnswer','iCatchAnswer']:
        dfQandA[intCol] = dfQandA[intCol].astype(int)

    # Compile demo q's
    if 'genderSlider.response' in dfIn.columns: # COVID01, Expectation, Stability
        # location
        try:
            location = GetNonNanValues(dfIn,'location (City, State)',False)[0]
        except:
            print('Location not found in file. Defaulting to blank.')
            location = ''
        dfQandA.loc[0,'Question'] = 'location'
        dfQandA.loc[0,'Answer'] = location
        # gender
        dfQandA.loc[1,'Question'] = 'gender'
        dfQandA.loc[1,'iAnswer'] = int(GetNonNanValues(dfIn,'genderSlider.response',True))
        resps = ['Male','Female','Other']
        dfQandA.loc[1,'Answer'] = resps[dfQandA.loc[1,'iAnswer']-1]
        dfQandA.loc[1,'RT'] = GetNonNanValues(dfIn,'genderSlider.rt',True)
        # age
        dfQandA.loc[2,'Question'] = 'age'
        dfQandA.loc[2,'iAnswer':'Answer'] = GetNonNanValues(dfIn,'ageSlider.response',True)
        dfQandA.loc[2,'RT'] = GetNonNanValues(dfIn,'ageSlider.rt',True)
        # status
        dfQandA.loc[3,'Question'] = 'status'
        try:
            iRung = int(GetNonNanValues(dfIn,'ladderMouse.clicked_name',False)[0][1:])
            dfQandA.loc[3,'RT'] = GetNonNanValues(dfIn,'ladderMouse.time',False)[0]
        except IndexError:
            iRung = int(GetNonNanValues(dfIn,'ladderKeyResp.keys',True))
        dfQandA.loc[3,'iAnswer':'Answer'] = iRung
    else:
        # read demographics info from a dataCheck file
        dfDataCheck01 = pd.read_csv(demoDataFile, index_col=0)
        isMatch = (dfDataCheck01['MTurkID'] == mTurkID)
        # fill in
        demographics = ['gender','age','status']
        for iDemo,demo in enumerate(demographics):
            dfQandA.loc[iDemo,'Question'] = demo
            if demo in dfDataCheck01.columns:
                dfQandA.loc[iDemo,'Answer'] = dfDataCheck01.loc[isMatch,demo].values
            else:
                print('Location not found in file. Defaulting to blank.')
                dfQandA.loc[iDemo,'Answer'] = ''

    # Compile questionnaire q's
    for iPage in range(nPages):
        isThisPage = dfIn['Page']==(iPage+1)
        resps = dfIn.loc[isThisPage,'Responses'].values[0].split(',')
        for iQ in range(6):
            iLine = 6*iPage+iQ+nDemo
            dfQandA.loc[iLine,'Page'] = iPage+1
            if pd.notna(dfIn.loc[isThisPage,'slider%d.response'%(iQ+1)].values):
                dfQandA.loc[iLine,'Question'] = dfIn.loc[isThisPage,'Quest%d'%(iQ+1)].values
                dfQandA.loc[iLine,'iAnswer'] = int(dfIn.loc[isThisPage,'slider%d.response'%(iQ+1)].values)
                dfQandA.loc[iLine,'Answer'] = resps[int(dfQandA.loc[iLine,'iAnswer']-1)]
                dfQandA.loc[iLine,'RT'] = dfIn.loc[isThisPage,'slider%d.rt'%(iQ+1)].values
            else:
                dfQandA.loc[iLine,'Question'] = ''
                dfQandA.loc[iLine,'iAnswer'] = np.nan
                dfQandA.loc[iLine,'Answer'] = ''
                dfQandA.loc[iLine,'RT'] = np.nan


    # Compile COVID q's
    covidCols = [x.split('.')[0] for x in dfIn.columns if (x.startswith('s_covid19') and x.endswith('.response'))];
    iBoxAnswer = []
    iLine = 6*nPages+nDemo-1
    for col in covidCols:
        resp = GetNonNanValues(dfIn,col+'.response',True)
        page = dfIn.loc[pd.notna(dfIn[col+'.response']),'covidPage'].values[0]
        colID = col.split('_')[2]
        if colID=='13': # boxes
            iBox = int(col.split('.')[0][-1]) # get box number
            if resp==True:
                iBoxAnswer = iBoxAnswer + [iBox] # add this number to the list of checks
            if iBox==3: # for first one only, increment line number (to avoid overwriting)
                iLine = iLine + 1;
                resp = np.nan;
                RT = np.nan;
                impacts = np.array(['lost_job','physically_ill','mental_illness','no_change'])
                answer = str(impacts[iBoxAnswer])[1:-1]
                col = col[:-5] # cut out the _box3 suffix
            else:
                continue; # skip the adding of results
        else:
            iLine = iLine + 1;
            RT = GetNonNanValues(dfIn,col+'.RT',True)
            answer = '?' # TODO: read from condition files? Include in data file?

        # Add results to dfQA
        dfQandA.loc[iLine,'Page'] = page
        dfQandA.loc[iLine,'Question'] = col
        dfQandA.loc[iLine,'iAnswer'] = resp
        dfQandA.loc[iLine,'Answer'] = answer;
        dfQandA.loc[iLine,'RT'] = RT


    # Manually categorize
    dfQandA['Survey'] = 'BLANK'
    dfQandA.loc[0:4,'Survey'] = 'DEMOG'
    dfQandA.loc[4:24,'Survey'] = 'CESD'
    dfQandA.loc[28:41,'Survey'] = 'SHAPS'
    dfQandA.loc[[24,25,42],'Survey'] = 'CATCH'
    dfQandA.loc[46:,'Survey'] = 'COVID'
    # fill in catch answers manually
    dfQandA['iCatchAnswer'] = np.nan
    dfQandA.loc[[24,25,42],'iCatchAnswer'] = [2,1,3]

    for survey in np.unique(dfQandA.Survey):
        dfQandA.loc[dfQandA.Survey==survey,'SurveyQNum'] = np.arange(np.sum(dfQandA.Survey==survey))+1

    isCatch = dfQandA.Survey=='CATCH'
    catchCorrect = np.sum(dfQandA.loc[isCatch,'iAnswer']==dfQandA.loc[isCatch,'iCatchAnswer'])
    # print('participant %s: %d/%d catch questions correct'%(participant,catchCorrect,np.sum(isCatch)))

    return dfQandA
