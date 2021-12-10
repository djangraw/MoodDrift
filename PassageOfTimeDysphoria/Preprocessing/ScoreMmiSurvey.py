#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Score clinical and COVID surveys associated with the MMI task.

Created 4/21/20 by DJ.
Updated 12/3/21 by DJ - added MW and boredom scale scoring
"""
import pandas as pd
import numpy as np

def ScoreMmiSurvey(dfQandA,participant):
    # Read in .csv file
    cols = ['participant','gender','age','status','CESD','SHAPS','CATCH','COVID','CovidImpact']
    dfSurvey = pd.DataFrame(np.zeros((1,len(cols))),columns=cols);
    dfSurvey['CovidImpact'] = ''

    # add gender
    dfSurvey['participant'] = participant
    for demog in ['gender','age','status']:
        dfSurvey[demog] = dfQandA.loc[dfQandA.Question==demog,'Answer'].values;
    # score CESD
    plusQs = np.array([1,2,3,5,6,7,9,10,11,13,14,15,17,18,29,20])
    minusQs = np.array([4, 8, 12, 16])
    cesdTotal = np.sum(dfQandA.loc[(dfQandA.Survey=='CESD') & (dfQandA.SurveyQNum.isin(plusQs)),'iAnswer'].values - 1 )
    cesdTotal += np.sum(4 - dfQandA.loc[(dfQandA.Survey=='CESD') & (dfQandA.SurveyQNum.isin(minusQs)),'iAnswer'].values )
    dfSurvey['CESD'] = cesdTotal
    # score SHAPS
    dfSurvey['SHAPS'] = np.sum(dfQandA.loc[dfQandA.Survey=='SHAPS','iAnswer'].values <= 2)
    # score catch Qs
    isCatch = dfQandA.Survey=='CATCH'
    dfSurvey['CATCH'] = np.sum(dfQandA.loc[isCatch,'iAnswer']==dfQandA.loc[isCatch,'iCatchAnswer'])
    # score COVID
    iCovid = np.where(dfQandA.Survey=='COVID')[0]
    covidScore = 0;
    for iQ in iCovid:
        qID = dfQandA.loc[iQ,'Question'].split('_')[2]
        if qID in ['1','2','3a','4a','4b','4c','4d','5','6','7','8','16','17','18','19','20','21']: # scored 1-10
            covidScore = covidScore + dfQandA.loc[iQ,'iAnswer'] + 1
        elif qID in ['3b','3c','9','10','14','15']:
            covidScore = covidScore + 10 - dfQandA.loc[iQ,'iAnswer']
        elif qID in ['11','12a','12b']:
            covidScore = covidScore + int(dfQandA.loc[iQ,'iAnswer']==0) # 0=YES, 1=NO
        elif qID=='13':
            if str(dfQandA.loc[iQ,'Answer'])=='nan': # no response: interpret as no change
                print('subj %s gave no response for CovidImpact Q. Interpreting as no_change.'%dfSurvey['participant'])
                dfSurvey['CovidImpact'] = "'no_change'"
            else:
                dfSurvey['CovidImpact'] = dfQandA.loc[iQ,'Answer'] # string
        else:
            raise ValueError('COVID question ID %s not recognized... cannot score!'%qID)

    dfSurvey['COVID'] = covidScore
    # score MW
    dfSurvey['MW'] = np.sum(dfQandA.loc[dfQandA.Survey=='MW','iAnswer'].values)
    # score Boredom
    dfSurvey['BORED'] = np.sum(dfQandA.loc[dfQandA.Survey=='BORED','iAnswer'].values)

    dfSurvey['age'] = dfSurvey['age'].astype(float)
    dfSurvey['status'] = dfSurvey['status'].astype(float)

    # return result
    return dfSurvey;
