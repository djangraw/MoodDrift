
"""
RunPymerOnCovidData_Aug2020.py

Runs Pymer's linear mixed effects (LME) analysis on specified cohorts.

- Created Aug 2020 by DJ
- Updated 2/4/2021 by DJ - accommodated GbeConfirm cohort.
- Updated 3/31/21 by DJ - adapted for shared code structure.
- Updated 4/15/21 by DJ - added have_gbe flag, removed extraneous imports
"""

# %% import some basic libraries
import pandas as pd
import numpy as np
# import dataviz
import seaborn as sns
import matplotlib.pyplot as plt
# Import the linear regression model class
from pymer4.models import Lmer

# %% Declare constants
print('=== Loading Data... ===')

# Declare cohorts to run
have_gbe = False # are the Rutledge Great Brain Experiment data included?
cohortsToRun = ['AllOpeningRestAndRandom','COVID01','COVID02','COVID03','Stability01-Rest','Stability01-Rest_block2','Stability02-Rest']#,'Stability01-RandomVer2']
if have_gbe:
    cohortsToRun = cohortsToRun + ['GbeExplore','GbeConfirm'] # add on GBE cohorts

# Declare other parameters
procDataDir = '../Data/OutFiles' # path to preprocessed data
outDir = '../Data/OutFiles' # where results should be saved
outFigDir = '../Figures' # where figures should be saved
includeRepeats = False; # include cohorts that are not naive in the data
includeRelativeBaseline = False; # include relative baseline as a fixed effect in the LME
includeCohort = False; # include cohort as a random effect in the LME
mfqCutoff = 12.0 # Clinical cutoff of MFQ score to be classified as 'at risk'
cesdCutoff = 16.0 # Clinical cutoff of CES-D score to be classified as 'at risk'

# %% Run analysis
for outName in cohortsToRun:

    # Use cohort names to determine which data should be used
    if outName.endswith('_block2'):
        iBlock = 2;
    else:
        iBlock = 0
    ageCutoffs = [0, 16, 18, 40, 100] # defining age groups

    if outName.startswith('Gbe'):
        dfBatch = pd.DataFrame.from_dict({outName: ['%s/Mmi-%s_Ratings.csv'%(procDataDir,outName),
                                                         '%s/Mmi-%s_Survey.csv'%(procDataDir,outName),
                                                         '%s/Mmi-%s_Trial.csv'%(procDataDir,outName),
                                                         'random',0]},
                                        orient='index',columns=['ratingsFile','surveyFile','trialFile',
                                                                'block0_type','nPreviousRuns'])
    elif outName.startswith('AllOpeningRestAndRandom'):
        dfBatches = pd.read_csv('%s/Mmi-Batches.csv'%procDataDir)

        dfBatch = dfBatches[['batchName','ratingsFile','surveyFile','trialFile','lifeHappyFile','block0_type','nPreviousRuns']]
        dfBatch = dfBatch.loc[(dfBatches.block0_type=='rest') | (dfBatches.block0_type=='random'),:]
        if not includeRepeats:
            dfBatch = dfBatch.loc[(dfBatch.nPreviousRuns==0),:]
        dfBatch = dfBatch.set_index('batchName')
        dfBatch = dfBatch.drop(['Stability01-random','Stability02-random','RecoveryNimh-run3'],axis=0,errors='ignore')

    elif outName == 'Stability01-Rest_block2':
        dfBatches = pd.read_csv('%s/Mmi-Batches.csv'%procDataDir)
        dfBatch = dfBatches[['batchName','ratingsFile','surveyFile','trialFile','lifeHappyFile','block0_type','nPreviousRuns']]
        dfBatch = dfBatch.loc[dfBatches.batchName=='Stability01-Rest',:]
        dfBatch = dfBatch.set_index('batchName')
    elif outName == 'Recovery(Instructed)1':
        dfBatches = pd.read_csv('%s/Mmi-Batches.csv'%procDataDir)
        dfBatch = dfBatches[['batchName','ratingsFile','surveyFile','trialFile','lifeHappyFile','block0_type','nPreviousRuns']]
        dfBatch = dfBatch.loc[(x in ['Recovery1','RecoveryInstructed1'] for x in dfBatches.batchName),:]
        dfBatch = dfBatch.set_index('batchName')
    else:
        dfBatches = pd.read_csv('%s/Mmi-Batches.csv'%procDataDir)
        dfBatch = dfBatches[['batchName','ratingsFile','surveyFile','trialFile','lifeHappyFile','block0_type','nPreviousRuns']]
        dfBatch = dfBatch.loc[dfBatches.batchName==outName,:]
        dfBatch = dfBatch.set_index('batchName')

    # get batch names
    batchNames = dfBatch.index.values

    # %% Load data and assemble LME inputs

    # Load data
    dfRatingList = []
    iSubjAll = -1;
    for batchName,row in dfBatch.iterrows():
        # Get experiment type
        isTask = (row.block0_type!='rest')
        isGbe = ('Gbe' in batchName)
        print('Loading %s...'%row.ratingsFile)
        dfRating = pd.read_csv(row.ratingsFile);
        if isGbe:
            summaryFile = '%s/Mmi-GBE_Summary.csv'%(procDataDir)
            print('Loading %s...'%summaryFile)
            dfSurvey = pd.read_csv(summaryFile).set_index('participant')
        else:
            print('Loading %s...'%row.surveyFile)
            dfSurvey = pd.read_csv(row.surveyFile).set_index('participant');
            print('Loading %s...'%row.lifeHappyFile)
            dfLifeHappy = pd.read_csv(row.lifeHappyFile).set_index('participant');
            dfSurvey['lifeHappy'] = dfLifeHappy['rating']
        if isTask:
            print('Loading %s...'%row.trialFile)
            dfTrial = pd.read_csv(row.trialFile);
        else:
            dfRating['totalWinnings'] = 0
            dfRating['meanRPE'] = 0

        # Add cohort column
        dfRating['cohort'] = batchName
        # Fill in missing RTs with max rating time (same across all NIMH cohorts)
        dfRating.loc[np.isnan(dfRating.RT),'RT'] = 4.5
        # get particiapants list
        if 'Nimh-run1' in batchName:
            print('Making participant numbers in batch %s negative to avoid overlap with MTurk subjects.'%batchName)
            dfRating.participant = -dfRating.participant # make negative to avoid overlap with MTurk participant numbers
            dfSurvey.index = -dfSurvey.index # make negative to avoid overlap with MTurk participant numbers
            if isTask:
                dfTrial.participant = -dfTrial.participant # make negative to avoid overlap with MTurk participant numbers
        elif 'Nimh-run2' in batchName:
            print('Making participant numbers in batch %s negative and -900000 to avoid overlap with MTurk subjects.'%batchName)
            dfRating.participant = -dfRating.participant-900000 # make negative to avoid overlap with MTurk participant numbers
            dfSurvey.index = -dfSurvey.index-900000 # make negative to avoid overlap with MTurk participant numbers
            if isTask:
                dfTrial.participant = -dfTrial.participant-900000 # make negative to avoid overlap with MTurk participant numbers
        elif 'Nimh-run3' in batchName:
            print('Making participant numbers in batch %s negative and -9900000 to avoid overlap with MTurk subjects.'%batchName)
            dfRating.participant = -dfRating.participant-9900000 # make negative to avoid overlap with MTurk participant numbers
            dfSurvey.index = -dfSurvey.index-9900000 # make negative to avoid overlap with MTurk participant numbers
            if isTask:
                dfTrial.participant = -dfTrial.participant-9900000 # make negative to avoid overlap with MTurk participant numbers

        print('Adding columns to batch %s...'%batchName)
        participants = np.unique(dfRating.participant)
        # Subtract block start time and add iRating column so we can split early and late stages
        dfRating['iRating'] = np.nan
        isRightBlock = (dfRating.iBlock==iBlock)
        for participant in participants:
            # get indices of this subject and this subject+block
            isThisSubj = dfRating.participant==participant
            isThisBlock = isThisSubj & isRightBlock
            # get duration of mood question
            moodQuestionDur = dfRating.loc[isThisSubj,'time'].values[0] - dfRating.loc[isThisSubj,'RT'].values[0]
            # get start time of block (=start time of mood question)
            t0_block = dfRating.loc[isThisBlock,'time'].values[0] - dfRating.loc[isThisBlock,'RT'].values[0] - moodQuestionDur
            # subtract start time of block
            dfRating.loc[isThisBlock,'time'] = dfRating.loc[isThisBlock,'time'] - t0_block
            # Add iRating column
            dfRating.loc[isThisBlock,'iRating'] = np.arange(np.sum(isThisBlock))

            if not isGbe:
                dfRating.loc[isThisBlock,'isMale'] = dfSurvey.loc[participant,'gender']=='Male'
    #            dfRating.loc[isThisBlock,'ageOver40'] = dfSurvey.loc[participant,'age']-40
    #            dfRating.loc[isThisBlock,'isAdolescent'] = batchName.startswith('RecoveryNimh')
    #            dfRating.loc[isThisBlock,'isAdolescentXAgeOver15'] = float(batchName.startswith('RecoveryNimh')) * (dfSurvey.loc[participant,'age']-15)
                dfRating.loc[isThisBlock,'age'] = dfSurvey.loc[participant,'age']
                for iAge in range(len(ageCutoffs)-1):
                    dfRating.loc[isThisBlock,'isAge%dto%d'%(ageCutoffs[iAge],ageCutoffs[iAge+1])] = \
                        (dfSurvey.loc[participant,'age']>=ageCutoffs[iAge]) & \
                        (dfSurvey.loc[participant,'age']<ageCutoffs[iAge+1])

                if 'MFQ' in dfSurvey.columns:
                    dfRating.loc[isThisBlock,'isAtRisk'] = dfSurvey.loc[participant,'MFQ']>mfqCutoff
                    dfRating.loc[isThisBlock,'fracRiskScore'] = dfSurvey.loc[participant,'MFQ']/mfqCutoff
                elif 'CESD' in dfSurvey.columns:
                    dfRating.loc[isThisBlock,'isAtRisk'] = dfSurvey.loc[participant,'CESD']>cesdCutoff
                    dfRating.loc[isThisBlock,'fracRiskScore'] = dfSurvey.loc[participant,'CESD']/cesdCutoff

                else:
                    print('Warining: %s has neither MFQ nor CESD'%batchName)
                    dfRating.loc[isThisBlock,'isAtRisk'] = False;
                dfRating.loc[isThisBlock,'isRepeatParticipant'] = (row.nPreviousRuns>0)
            dfRating.loc[isThisBlock,'lifeHappyOver0p7'] = dfSurvey.loc[participant,'lifeHappy']-0.7
            dfRating.loc[isThisBlock,'meanIRIOver20'] = np.mean(np.diff(dfRating.loc[isThisBlock,'time']))-20
            dfRating.loc[isThisBlock,'medianIRIOver20'] = np.median(np.diff(dfRating.loc[isThisBlock,'time']))-20
            if isTask:
                dfRating.loc[isThisBlock,'totalWinnings'] = np.sum(dfTrial.loc[(dfTrial.participant==participant) & (dfTrial.iBlock==iBlock),'outcomeAmount'])
                dfRating.loc[isThisBlock,'meanRPE'] = np.mean(dfTrial.loc[(dfTrial.participant==participant) & (dfTrial.iBlock==iBlock),'RPE'])


        # Crop to block and columns of interest and Add to list
        if isGbe:
            dfRatingList.append(dfRating.loc[isRightBlock,['cohort','participant','rating',
                                        'time','iRating','meanIRIOver20',
                                        'totalWinnings','meanRPE','lifeHappyOver0p7']])
        else:
            if includeRepeats:
                dfRatingList.append(dfRating.loc[isRightBlock,['cohort','participant','rating',
                                        'time','iRating','isMale','ageOver40',
                                        'isAdolescent','isAdolescentXAgeOver15','fracRiskScore',
                                        'isRepeatParticipant','meanIRIOver20',
                                        'totalWinnings','meanRPE']])
            else:
    #            dfRatingList.append(dfRating.loc[isRightBlock,['cohort','participant','rating',
    #                                    'time','iRating','isMale','ageOver40',
    #                                    'isAdolescent','isAdolescentXAgeOver15','fracRiskScore',
    #                                    'meanIRIOver20','totalWinnings','meanRPE']])
                dfRatingList.append(dfRating.loc[isRightBlock,['cohort','participant','rating',
                                        'time','iRating','isMale','meanIRIOver20',
                                        'totalWinnings','meanRPE','fracRiskScore',
                                        'isAge0to16','isAge16to18','isAge40to100']])
    #            dfRatingList.append(dfRating.loc[isRightBlock,['cohort','participant','rating',
    #                                    'time','iRating','isMale','meanIRIOver20',
    #                                    'totalWinnings','meanRPE','fracRiskScore','age']])




    # %% Arrange into dataframe
    print('Building Dataframe...')
    dfAll = pd.concat(dfRatingList,axis=0)
    dfAll.columns = ['Cohort','Subject','Mood','Time'] + dfAll.columns[4:].tolist()
    dfAll['Subject'] = dfAll['Subject'].astype(int)

    nOld = dfAll.shape[0]
    # Remove NaNs
    dfAll = dfAll.loc[~np.any(pd.isna(dfAll),axis=1),:]
    nNew = dfAll.shape[0]
    pctRemoved = 100.0*(nOld-nNew)/nOld
    print('%.2f%% of data removed for containing NaNs.'%pctRemoved);

    # Remove outlier meanIRIs
    if isGbe and ('meanIRIOver20' in dfAll.columns):
        # detect outliers
        IRIs = np.sort(dfAll.meanIRIOver20)

        Q1 = np.percentile(IRIs,25)
        Q3 = np.percentile(IRIs,75)
        IQR = Q3-Q1
        lowCutoff = Q1-1.5*IQR
        highCutoff = Q3+1.5*IQR
        print('Excluding outliers based on Q1-1.5*IQR and Q3+1.5*IQR...')
        print('   Cutoffs: low = %.3g, high = %.3g'%(lowCutoff,highCutoff))

        pctUnder = np.mean(IRIs<lowCutoff)*100
        pctOver = np.mean(IRIs>highCutoff)*100
        print('   Excluding %.3g%% under low, %.3g%% over high, %.3g%% total'%(pctUnder,pctOver,pctUnder+pctOver))

        # Remove high & low outliers
        nOld = dfAll.shape[0]
    #    dfAll = dfAll.loc[(dfAll.meanIRIOver20<38),:]
        dfAll = dfAll.loc[(dfAll.meanIRIOver20<=highCutoff) & (dfAll.meanIRIOver20>=lowCutoff),:]
        nNew = dfAll.shape[0]
        pctRemoved = 100.0*(nOld-nNew)/nOld
    #    print('%.2f%% of data removed for having meanIRI>38.'%pctRemoved);
        print('   %.2f%% of data removed for having outlier meanIRIs.'%pctRemoved);


    # add column for all but first cohort
    #for batchName in batchNames[1:]:
    #    dfAll['isCohort%s'%batchName] = (dfAll.Cohort==batchName)

    # z-score time variable
    #dfAll['Time'] = (dfAll['Time'] - dfAll['Time'].mean())/dfAll['Time'].std()

    # divide by 60 to get time in minutes
    dfAll['Time'] = dfAll['Time']/60

    if includeRelativeBaseline:
        # Add term for baseline - grp baseline (to capture regression to the mean)
        print('Adding term for relative baseline (to capture regression to the mean)...')
        subjects = np.unique(dfAll.Subject)
        baseline = np.zeros(len(subjects))
        for iSubj,subj in enumerate(subjects):
            baseline[iSubj] = dfAll.loc[dfAll.Subject == subj, 'Mood'].values[0]

        meanBaseline = np.mean(baseline)
        print('   Mean baseline = %g'%meanBaseline)
        for iSubj,subj in enumerate(subjects):
            dfAll.loc[dfAll.Subject == subj,'RelativeBaseline'] = baseline[iSubj] - meanBaseline

    # Print input data
    print(dfAll.head())

    # %% Declare pymer model

    if outName.startswith('Gbe'): # Model for Mobile App participants
        lmString = 'Mood ~ 1 + Time * (meanIRIOver20 + totalWinnings + meanRPE + lifeHappyOver0p7) + (Time | Subject)'
    elif includeRelativeBaseline: # Online participants if each cohort's baseline should be included
        lmString = 'Mood ~ 1 + RelativeBaseline + Time * (' + ' + '.join(dfAll.columns[5:-1]) + ') + (Time | Subject)'
    else: # Online participants
        lmString = 'Mood ~ 1 + Time * (isMale + meanIRIOver20 + totalWinnings + meanRPE + fracRiskScore + isAge0to16 + isAge16to18 + isAge40to100) + (Time | Subject)'
        # Alternative models
#        lmString = 'Mood ~ 1 + Time * (' + ' + '.join(dfAll.columns[5:]) + ') + (Time | Subject)'
#        lmString = 'Mood ~ 1 + isMale + ageOver40 + isAdolescent + isAdolescentXAgeOver15 + meanIRIOver20 + totalWinnings + meanRPE + Time * fracRiskScore + (Time | Subject)'
#        lmString = 'Mood ~ 1 + isMale + ageOver40 + isAdolescent + meanIRIOver20 + totalWinnings + meanRPE + Time * (fracRiskScore + isAdolescentXAgeOver15) + (Time | Subject)'
#        lmString = 'Mood ~ 1 + isMale + meanIRIOver20 + totalWinnings + meanRPE + Time * (fracRiskScore + ageOver40 + isAdolescent + isAdolescentXAgeOver15) + (Time | Subject)'

    # Alternative models
    #for column in dfAll.columns[5:]:
    #    lmString = lmString + ' + %s'%column
    #lmString = lmString + ') + (Time | Subject)'
    #lmString = 'Mood ~ 1 + Time * (' + ' + '.join(dfAll.columns[5:]) + ') + Time*ageOver40*isAtRisk + (Time | Subject)'
    #lmString = 'Mood ~ 1 + Time*(isMale + isAdolescent + (isAdolescent * isAtRisk) + isRepeatParticipant + meanIRIOver20 + totalWinnings + meanRPE) + (Time|Subject)'

    if includeCohort:
        lmString = lmString + ' + (Time | Cohort)'
    print('=== LM Model: ===')
    print(lmString)


    # %% Fit Pymer Model


    for stage in ['full','late']: # full = all ratings, early = use only first 2 ratings, late = exclude first rating
    #for stage in ['full','early','late']:

        print('===== RUNNING %s MODEL ====='%stage.upper())
        # %% Initialize model
        print('=== Setting up Model... ===')
        if stage=='early':
            # prepare input
            dfData = dfAll.loc[dfAll.iRating<=1,:] # trials 0-1
            # simulate LME model fit output
            subjects = np.unique(dfData.Subject)
            dfFixef = pd.DataFrame(np.zeros((len(subjects),2)),index=subjects,columns=['(Intercept)','Time'])
            for subj in subjects:
                is0 = (dfData.Subject==subj) & (dfData.iRating==0)
                is1 = (dfData.Subject==subj) & (dfData.iRating==1)
                m = (dfData.loc[is1,'Mood'].values - dfData.loc[is0,'Mood'].values) / \
                    (dfData.loc[is1,'Time'].values-dfData.loc[is0,'Time'].values)
                b = dfData.loc[is0,'Mood'].values - m * dfData.loc[is0,'Time'].values
                dfFixef.loc[subj,'(Intercept)'] = b
                dfFixef.loc[subj,'Time'] = m

        else:
            # Prepare input
            if stage=='late':
                dfData = dfAll.loc[dfAll.iRating>=1,:].copy() # trials 1-end
            elif stage=='full':
                dfData = dfAll
            else:
                raise ValueError('Stage %s not recognized!'%stage)

            # cast bool columns to bool
            for cc in dfData.columns:
                colvals = sorted(dfData[cc].unique())
                boolcol = True
                for cv in colvals:
                    if (not isinstance(cv, bool)) and (not isinstance(cv, np.bool_)):
                        boolcol = False
                if boolcol:
                    dfData[cc] = dfData[cc].astype(bool)

            # fit LME model
            model = Lmer(lmString,data=dfData)
            
            # Fit it
            print('=== Fitting Model... ===')
            dfFit = model.fit()
            print(dfFit)

            # Print model AIC
            print('=== Printing Results... ===')
            print('AIC = %d'%model.AIC)

            # Look at model data, including residuals
            print(model.data.head())

            # Get fixed FX output
            dfFixef = model.fixef

            # print fit values for a few subjects
            try:
                print(dfFixef.head(5))
            except:
                print('Could not print dfFixef.')

            # plot model predicted values against true values
            print('=== Plotting Predictions... ===')
            if 'fits' in model.data.columns:
                fig = plt.figure()
                sns_plot = sns.regplot(x='fits', y='Mood', data=model.data, fit_reg=True)
                outFile = '%s/Mmi-%s_pymerFits-%s.png'%(outFigDir,outName,stage)
                print('Saving %s...'%outFile)
                plt.savefig(outFile)
            else:
                print('model.data did not have fits column... skipping plot.')

            print('=== Saving Model Results... ===')
            # outFile = '%s/Mmi-%s_pymerModel-%s.h5'%(outDir,outName,stage)
            # print('Saving %s...'%outFile)
            # save_model(model, outFile)

            outFile = '%s/Mmi-%s_pymerFit-%s.csv'%(outDir,outName,stage)
            print('Saving %s...'%outFile)
            dfFit.to_csv(outFile,float_format='%.6f')

        # Save results common to all stages
        print('=== Saving Subject Slopes+Intercepts... ===')
        if includeCohort and stage!='early': # 2 random effects => dfFixef is a list
            for i,label in enumerate(['Subject','Cohort']):
                outFile = '%s/Mmi-%s_pymerCoeffs-%s-%s.csv'%(outDir,outName,label,stage)
                print('Saving %s...'%outFile)
                dfFixef[i].to_csv(outFile,index_label=label,float_format='%.6f')
        else:
            outFile = '%s/Mmi-%s_pymerCoeffs-%s.csv'%(outDir,outName,stage)
            print('Saving %s...'%outFile)
            dfFixef.to_csv(outFile,index_label='Subject',float_format='%.6f')
        # Save model input
        outFile = '%s/Mmi-%s_pymerInput-%s.csv'%(outDir,outName,stage)
        print('Saving %s...'%outFile)
        dfData.to_csv(outFile,float_format='%.6f')

        print('=== Done! ===')
