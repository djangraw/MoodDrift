#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Produce all results for the Passage-of-Time Dysphoria paper.
To use, run whole script or cell-by-cell for specific results.

- Created 10/22/20 by DJ.
- Updated 3/31/21 by DJ - adapted for shared code structure.
- Updated 5/6/21 by DJ - added code to produce several figures found in paper
"""

# Import packages
import PassageOfTimeDysphoria.Analysis.PlotMmiData as pmd
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from PassageOfTimeDysphoria.Analysis.CompareMmiRatings import CompareMmiRatings
import PassageOfTimeDysphoria.Analysis.PlotPytorchPenaltyTuning as ppt
from PassageOfTimeDysphoria.Analysis.CalculatePytorchModelError import CalculatePytorchModelError
from PassageOfTimeDysphoria.Analysis.PlotAgeVsCoeffs import PlotAgeVsCoeffs
from PassageOfTimeDysphoria.Analysis.PlotTimeOfDayVsSlopeAndIntercept import PlotTimeOfDayVsSlopeAndIntercept
from PassageOfTimeDysphoria.Analysis.PlotPymerFits import PlotPymerHistosJoint
import PassageOfTimeDysphoria.Analysis.GetMmiIcc as gmi
from scipy import stats
import seaborn as sns

# Use exploratory (True) or Confirmatory (False) mobile app participants?
IS_EXPLORE = False # GbeExplore (True) or GbeConfirm (False)
dataDir = '../Data/OutFiles' # path to processed data
pytorchDir = '../Data/GbePytorchResults' # path to model fitting results
outFigDir = '../Figures' # where model fitting figures should be saved
have_gbe = True

# %% Print Cohen's D for original cohort and others
first_and_lasts = []
for batchName in ['Recovery(Instructed)1', 'AdultOpeningRest', 'RecoveryNimh-run1','AllOpeningRestAndRandom']:

    dfRating = pd.read_csv('%s/Mmi-%s_Ratings.csv'%(dataDir,batchName), index_col=0)
    #dfMeanRating = pmd.GetMeanRatings(dfRating.loc[dfRating.iBlock==0,:],nRatings=-1,participantLabel='mean')
    nSubj = len(np.unique(dfRating.participant))

    first_trial = dfRating.loc[(dfRating.iBlock == 0)].groupby('participant').first().reset_index()
    last_trial = dfRating.loc[dfRating.iBlock == 0].groupby('participant').last().reset_index()
    first_and_last = first_trial.merge(last_trial, how='left', on='participant', suffixes=['_first', '_last'])

    first_and_last['dif'] = first_and_last.rating_last - first_and_last.rating_first
    first_and_last['time_dif'] = first_and_last.time_last - first_and_last.time_first
    first_and_last['batch'] = batchName
    first_and_lasts.append(first_and_last)
    M0 = first_and_last.rating_first.mean()
    M1 = first_and_last.rating_last.mean()
    SD0 = first_and_last.rating_first.std()
    SD1 = first_and_last.rating_last.std()
    t1 = (first_and_last.time_last - first_and_last.time_first).mean()/60

    SDpooled = np.sqrt((SD0**2+SD1**2)/2)
    cohensD = (M1-M0)/SDpooled

    md_se = first_and_last.dif.std()/np.sqrt(len(first_and_last))

    print(f"Batch {batchName} (n={nSubj}): After {t1:.1f} minutes, difference is {(M1-M0)*100:0.2f} +- {md_se*100:0.2f},  Cohen's D = {cohensD:.3g}")

first_and_lasts = pd.concat(first_and_lasts)
adult_difs = first_and_lasts.loc[first_and_lasts.batch == 'AdultOpeningRest'].dif
adolescent_difs = first_and_lasts.loc[first_and_lasts.batch == 'RecoveryNimh-run1'].dif
stat, pvalue = stats.ttest_ind(adult_difs, adolescent_difs)
print(f"Difference between adult and adolescent: n = {len(adult_difs) + len(adolescent_difs)} dof = {len(adult_difs) + len(adolescent_difs) - 2}, stat= {stat:.3g}, pvalue= {pvalue:.3g}")


# %% Plot all naive opening rest batches separately
batchNames = ['Recovery(Instructed)1', 'Expectation-7min','Expectation-12min','RestDownUp','Stability01-Rest','COVID01']
batchLabels = ['15sRestBetween','Expectation-7mRest','Expectation-12mRest','RestDownUp','Daily-Rest-01','Weekly-Rest-01']
CompareMmiRatings(batchNames,batchLabels=batchLabels,iBlock=0,doInterpolation=True)
# Annotate plot
plt.title('Passage-of-Time dysphoria persists across all MTurk cohorts receiving opening rest')
plt.gca().set_axisbelow(True)
plt.ylim([0.4,0.8])
plt.grid()
# Save figure
outFig = '%s/Mmi_%s_Comparison.png'%(outFigDir,'-'.join(batchNames))
print('Saving figure as %s...'%outFig)
plt.savefig(outFig)
print('Done!')


# %% Plot online adult vs. in-person adolescent cohort


batchNames = ['AdultOpeningRest','RecoveryNimh-run1']
batchLabels = ['MTurk cohorts','In-person adolescent cohort']
CompareMmiRatings(batchNames,batchLabels=batchLabels,iBlock=0,doInterpolation=True)
# Annotate plot
plt.title('Passage-of-Time dysphoria generalizes to different age group & recruitment method')
plt.gca().set_axisbelow(True)
plt.ylim([0.4,0.8])
plt.grid()
# Save figure
outFig = '%s/Mmi_%s_Comparison.png'%(outFigDir,'-'.join(batchNames))
print('Saving figure as %s...'%outFig)
plt.savefig(outFig)
print('Done!')

# %% Plot simple task cohorts
batchNames = ['Recovery(Instructed)1','MotionFeedback','Stability01-RandomVer2']
#batchLabels = ['Rest','Visuomotor task','Random gambling']
batchLabels = ['15sRestBetween','Visuomotor-Feedback','Daily-Random-01']
CompareMmiRatings(batchNames,batchLabels=batchLabels,iBlock='all',doInterpolation=True)
# Annotate plot
plt.title('Passage-of-Time dysphoria persists in presence of simple tasks')
plt.gca().set_axisbelow(True)
plt.ylim([0.4,0.8])
plt.xlim([-20,500])
plt.grid()
# Save figure
outFig = '%s/Mmi_%s_Comparison.png'%(outFigDir,'-'.join(batchNames))
print('Saving figure as %s...'%outFig)
plt.savefig(outFig)
print('Done!')


# %% LME results: Mean decline and Cohen's D with time

# Load LME results
batchName = 'AllOpeningRestAndRandom'
stage = 'full'
inFile = '%s/Mmi-%s_pymerCoeffs-%s.csv'%(dataDir,batchName,stage)
print('Loading pymer fits from %s...'%inFile)
dfCoeffs = pd.read_csv(inFile)
print('Done!')

# Cohen's D for LME results
moodSlope = dfCoeffs.Time
mood10 = moodSlope*10.0 # decline in mood after 10 minutes
D = np.mean(mood10)/np.std(mood10)
_,p = stats.wilcoxon(moodSlope)
print('===LME RESULTS FOR ONLINE PARTICIPANTS:===')
print('Decline in mood = %.3g +/- %.3g %%/min'%(np.mean(moodSlope*100),np.std(moodSlope*100)/np.sqrt(moodSlope.size)))
print('Decline in mood after 10 minutes: %.3g%% +/- %.3g, Cohen''s D=%.3g'%(np.mean(mood10)*100, np.std(mood10*100)/np.sqrt(mood10.size),D))
print('Wilcoxon signed rank vs. 0: p=%.3g'%p)
slope_range = dfCoeffs.Time.quantile([0.025,0.975]).values*100
print(f"2.5percentile slope = {slope_range[0]:0.3f}, 97.5percentile slope = {slope_range[1]:0.3f}")

# %% Test difference between adolescents and not
isAdolescent = dfCoeffs.Subject<0
T,p = stats.ttest_ind(dfCoeffs.loc[isAdolescent,'Time'],dfCoeffs.loc[~isAdolescent,'Time'])
print('Adolescents vs. not: T=%.3g, p=%.3g'%(T,p))

# %% Get impacts of gender, IRI, winnings, & RPEs from the LME results table
batchName = 'AllOpeningRestAndRandom'
stage = 'full'
inFile = '%s/Mmi-%s_PymerFit-%s.csv'%(dataDir,batchName,stage)
print('Loading pymer fits from %s...'%inFile)
dfFits = pd.read_csv(inFile,index_col=0)
print('Done!')


m = dfFits.loc['Time:isMaleTRUE','Estimate']*100
se = dfFits.loc['Time:isMaleTRUE','SE']*100
T = dfFits.loc['Time:isMaleTRUE','T-stat']
dof = dfFits.loc['Time:isMaleTRUE', 'DF']
p = dfFits.loc['Time:isMaleTRUE','P-val']
print('Gender x slope in LME:')
print('%.3g +/- %.3g %% mood, T=%.3g, dof=%0.3g, p=%.3g'%(m,se,T,dof,p))

m = dfFits.loc['Time:meanIRIOver20','Estimate']*100
se = dfFits.loc['Time:meanIRIOver20','SE']*100
T = dfFits.loc['Time:meanIRIOver20','T-stat']
dof = dfFits.loc['Time:meanIRIOver20', 'DF']
p = dfFits.loc['Time:meanIRIOver20','P-val']
print('Inter-Rating Interval x slope in LME:')
print('%.3g +/- %.3g %% mood, T=%.3g, dof=%0.3g, p=%.3g'%(m,se,T,dof,p))

m = dfFits.loc['Time:totalWinnings','Estimate']*100
se = dfFits.loc['Time:totalWinnings','SE']*100
T = dfFits.loc['Time:totalWinnings','T-stat']
dof = dfFits.loc['Time:totalWinnings', 'DF']
p = dfFits.loc['Time:totalWinnings','P-val']
print('Total Winnings x slope in LME:')
print('%.3g +/- %.3g %% mood, T=%.3g, dof=%0.3g, p=%.3g'%(m,se,T,dof,p))

m = dfFits.loc['Time:meanRPE','Estimate']*100
se = dfFits.loc['Time:meanRPE','SE']*100
T = dfFits.loc['Time:meanRPE','T-stat']
dof = dfFits.loc['Time:meanRPE', 'DF']
p = dfFits.loc['Time:meanRPE','P-val']
print('Mean RPE x slope in LME:')
print('%.3g +/- %.3g %% mood, T=%.3g, dof=%0.3g, p=%.3g'%(m,se,T,dof,p))


# %% Table02
print('Table 2 comes from %s.'%inFile)


# %% Plot mood over time with various IRIs

batchNames = ['RecoveryInstructed1Freq0p25','RecoveryInstructed1Freq0p5','Recovery(Instructed)1','RecoveryInstructed1Freq2']
#batchLabels = ['60 s rest between ratings','30 s rest between ratings','15 s rest between ratings','7.5 s rest between ratings']
batchLabels = ['60sRestBetween','30sRestBetween','15sRestBetween','7.5sRestBetween']
CompareMmiRatings(batchNames,batchLabels=batchLabels,iBlock=0,doInterpolation=True)
# Annotate plot
plt.title('Mood rating frequency does not affect passage-of-time dysphoria slope')
plt.gca().set_axisbelow(True)
plt.ylim([0.4,0.8])
plt.grid()
# Save figure
outFig = '%s/Mmi_%s_Comparison.png'%(outFigDir,'-'.join(batchNames))
print('Saving figure as %s...'%outFig)
plt.savefig(outFig)
print('Done!')



# %% Rating Method, Expectations, Task, Random Gambling

inFile = '%s/Mmi-%s_PymerCoeffs-%s.csv'%(dataDir,batchName,stage)
print('Loading pymer coefficients from %s...'%inFile)
dfCoeffs = pd.read_csv(inFile)
print('Done!')
inFile = '%s/Mmi-%s_PymerInput-%s.csv'%(dataDir,batchName,stage)
print('Loading pymer input from %s...'%inFile)
dfPymerIn = pd.read_csv(inFile,index_col=0)
dfPymerIn.loc[dfPymerIn.Cohort=='Recovery1','Cohort'] = 'Recovery(Instructed)1'
dfPymerIn.loc[dfPymerIn.Cohort=='RecoveryInstructed1','Cohort'] = 'Recovery(Instructed)1'


for batchNames in [['Numbers','Recovery(Instructed)1'],
                    ['Expectation-7min','Expectation-12min'],
                    ['MotionFeedback','Recovery(Instructed)1'],
                    ['Stability01-RandomVer2','Recovery(Instructed)1']]:

    cohort0 = np.unique(dfPymerIn.loc[dfPymerIn.Cohort==batchNames[0],'Subject'])
    isIn0 = [x in cohort0 for x in dfCoeffs.Subject]

    cohort1 = np.unique(dfPymerIn.loc[dfPymerIn.Cohort==batchNames[1],'Subject'])
    isIn1 = [x in cohort1 for x in dfCoeffs.Subject]

    T,p = stats.ttest_ind(dfCoeffs.loc[isIn0,'Time'],dfCoeffs.loc[isIn1,'Time'])
    n0 = np.sum(isIn0)
    n1 = np.sum(isIn1)
    dof = n0 + n1 - 2
    print('*** %s (n=%d) vs. %s (n=%d): T=%.3g, dof=%.3g p=%.3g'%(batchNames[0],n0,batchNames[1],n1,T,dof,p))

if have_gbe:
    for is_late in [False,True]:
        # %% Pytorch: including beta_T improves fit to testing data
        CalculatePytorchModelError(IS_EXPLORE, IS_LATE=is_late, dataDir = dataDir, pytorchDir = pytorchDir, outFigDir = outFigDir)

    # %% Plot penalty tuning

    for suffix in ['_tune-Oct2020', '_tune-noBetaT']:
        ppt.PlotPenaltyTuning(suffix,dataDir=pytorchDir,outFigDir=outFigDir)

    # %% Penalty tuning excluding first rating  (12/19/20)
    for suffix in ['_tune-late','_tune-late-noBetaT']:
        ppt.PlotPenaltyTuning(suffix,dataDir=pytorchDir,outFigDir=outFigDir)


    # %% Plot parameter distributions

    if IS_EXPLORE:
        suffix = '_GbeExplore'
    else:
        suffix = '_GbeConfirm'

    for stage in ['full','late']:
        if stage=='late':
            suffix = suffix + '-late'
        # Load results
        paramInFile = '%s/PyTorchParameters%s.csv'%(pytorchDir,suffix)
        print('Loading pyTorch best parameters from %s...'%paramInFile)
        best_pars = pd.read_csv(paramInFile,index_col=0).drop('participant',axis=1);
        params = best_pars.columns; # exclude lifeHappiness
        paramLabelDict = {'m0': r'$M_0$',
                       'lambda': r'$\lambda$',
                       'beta_E': r'$\beta_E$',
                       'beta_A': r'$\beta_A$',
                       'beta_T': r'$\beta_T$',
                       'SSE': 'SSE',
                       'lifeHappy':'life happiness'}
        print('Done!')

        # Add lifeHappy to best_pars and beta_T to dfSummary
        if IS_EXPLORE:
            summaryFile = '%s/Mmi-GbeExplore_Summary.csv'%(dataDir)
        else:
            summaryFile = '%s/Mmi-GbeConfirm_Summary.csv'%(dataDir)
        dfSummary = pd.read_csv(summaryFile,index_col=0)

        best_pars['lifeHappy'] = dfSummary['lifeHappy'].values
        dfSummary['beta_T'] = best_pars['beta_T'].values

        isTop = best_pars.lifeHappy>=np.median(best_pars.lifeHappy)

        # Plot parameter histograms
        plt.figure(264,figsize=(14,6)); plt.clf()
        nRows = 2
        nCols = 3
        for i,col in enumerate(params):
            # plot
            plt.subplot(nRows,nCols,i+1)
            plt.hist(best_pars[col],50)
            # annotate axis
            plt.xlabel(paramLabelDict[col])
            plt.ylabel('Number of subjects (n=%d)'%dfSummary.shape[0])
            plt.grid()

        # annotate figure
        plt.tight_layout(rect=(0,0,1.0,0.93))
        plt.suptitle('Computational model parameter fits')

        # save results
        outFile = '%s/PytorchParamHistos%s.png'%(outFigDir,suffix)
        print('Saving figure %s...'%outFile)
        plt.savefig(outFile)
        print('Done!')


    # %% Get stats on beta_T vs. 0

    for stage in ['full','late']:
        print('=== STAGE %s ==='%stage)
        # Load pytorch results
        if IS_EXPLORE:
            suffix = '_GbeExplore'
        else:
            suffix = '_GbeConfirm'
        if stage=='late':
            suffix = suffix + '-late'
        inFile = '%s/PyTorchParameters%s.csv'%(pytorchDir,suffix)
        print('Loading best parameters from %s...'%inFile)
        best_pars = pd.read_csv(inFile);

        #stat,p = stats.ttest_1samp(best_pars['beta_T'],0)
        print('mean +/- SE beta_T: %.3g%% mood/min +/- %.3g'%(np.mean(best_pars['beta_T'])*100,np.std(best_pars['beta_T'])*100/np.sqrt(best_pars.shape[0])))
        #print('2-tailed t-test on beta_T vs. 0: T=%.3g, p=%.3g'%(stat,p))

        stat,p = stats.wilcoxon(best_pars['beta_T'])
        #print('median beta_T: %.3g'%np.median(best_pars['beta_T']))
        print(f'2-sided wilcoxon sign-rank test on beta_T vs. 0: n={len(best_pars)}, dof={len(best_pars) - 1}, stat={stat:0.3g}, p={p:.3g}')
        print(f'stat in full {stat}')
    # %% Get stats on Mobile app LME slopes vs. 0

    batchName_online = 'AllOpeningRestAndRandom'
    if IS_EXPLORE:
        batchName_app = 'GbeExplore'
    else:
        batchName_app = 'GbeConfirm'

    for stage in ['full','late']:
        print('=== STAGE = %s ==='%stage)
        #dfPymerFit = pd.read_csv('%s/Mmi-%s_pymerFit-full.csv'%(dataDir,batchName),index_col=0)
        dfPymerCoeffs_online = pd.read_csv('%s/Mmi-%s_pymerCoeffs-%s.csv'%(dataDir,batchName_online,stage),index_col=0)
        dfPymerCoeffs_app = pd.read_csv('%s/Mmi-%s_pymerCoeffs-%s.csv'%(dataDir,batchName_app,stage),index_col=0)

        #stat,p = stats.ttest_1samp(best_pars['beta_T'],0)
        print('mean +/- SE LME slope param: %.3g%% mood/min +/- %.3g'%(np.mean(dfPymerCoeffs_app["Time"])*100,np.std(dfPymerCoeffs_app["Time"])*100/np.sqrt(dfPymerCoeffs_app.shape[0])))
        #print('2-tailed t-test on beta_T vs. 0: T=%.3g, p=%.3g'%(stat,p))

        stat,p = stats.wilcoxon(dfPymerCoeffs_app['Time'])
        #print('median beta_T: %.3g'%np.median(best_pars['beta_T']))
        print(f'2-sided wilcoxon sign-rank test on {batchName_app} LME slope vs. 0: n={len(dfPymerCoeffs_app["Time"])}, dof={len(dfPymerCoeffs_app["Time"]) - 1}, stat={stat:.3g}, p={p:.3g}')

        # Print ranksum comparison
        stat,p = stats.ranksums(dfPymerCoeffs_online.Time, dfPymerCoeffs_app.Time)
        nonline = len(dfPymerCoeffs_online.Time)
        napp = len(dfPymerCoeffs_app.Time)
        dof = nonline + napp - 2
        print(f'Ranksum of LME time coeff for online ({batchName_online}) vs. mobile app ({batchName_app}): nonline={nonline}, napp={napp}, ndof={dof}, stat={stat:.3g}, p={p:.3g}')




    # %% Compare LME and comp model
    # Load pytorch results
    if IS_EXPLORE:
        suffix = '_GbeExplore'
    else:
        suffix = '_GbeConfirm'
    inFile = '%s/PyTorchParameters%s.csv'%(pytorchDir,suffix)
    print('Loading best parameters from %s...'%inFile)
    best_pars = pd.read_csv(inFile);

    # Load LME results
    batchName = 'AllOpeningRestAndRandom'
    stage = 'full'
    inFile = '%s/Mmi-%s_pymerCoeffs-%s.csv'%(dataDir,batchName,stage)
    print('Loading pymer fits from %s...'%inFile)
    dfCoeffs = pd.read_csv(inFile)
    print('Done!')

    # Print ranksum comparison

    stat,p = stats.ranksums(dfCoeffs.Time, best_pars.beta_T)
    print('Ranksum of LME time coeff vs. PyTorch beta_T (%s): p=%.3g'%(suffix,p))


    # %% Plot histograms of LME slopes from online and mobile app data
    # Set up figure
    plt.close(632);
    plt.figure(632,figsize=(6,4),dpi=180, facecolor='w', edgecolor='k')
    plt.clf();

    batchName_online = 'AllOpeningRestAndRandom'
    if IS_EXPLORE:
        batchName_app = 'GbeExplore'
    else:
        batchName_app = 'GbeConfirm'
    #dfPymerFit = pd.read_csv('%s/Mmi-%s_pymerFit-full.csv'%(dataDir,batchName),index_col=0)
    dfPymerCoeffs_online = pd.read_csv('%s/Mmi-%s_pymerCoeffs-full.csv'%(dataDir,batchName_online),index_col=0)
    dfPymerCoeffs_app = pd.read_csv('%s/Mmi-%s_pymerCoeffs-full.csv'%(dataDir,batchName_app),index_col=0)

    # Plot histograms
    xHist = np.linspace(-10.0,10.0,100)
    nSubj_online = dfPymerCoeffs_online.shape[0]
    weights = np.ones(nSubj_online)/nSubj_online*100
    plt.hist(dfPymerCoeffs_online['Time']*100.0,xHist,weights=weights,alpha=0.5,label='All online participants (n=%d), LME'%nSubj_online)
    nSubj_app = dfPymerCoeffs_app.shape[0]
    weights = np.ones(nSubj_app)/nSubj_app*100
    if IS_EXPLORE:
        plt.hist(dfPymerCoeffs_app['Time']*100.0,xHist,weights=weights,alpha=0.5,label='Exploratory mobile app participants (n=%d), LME'%nSubj_app)
    else:
        plt.hist(dfPymerCoeffs_app['Time']*100.0,xHist,weights=weights,alpha=0.5,label='Confirmatory mobile app participants (n=%d), LME'%nSubj_app)

    # Annotate plot
    plt.grid(True)
    plt.xlabel('LME slope parameter (% mood/min)')
    plt.ylabel('Percent of participants')
    plt.legend()
    plt.title('LME mood slope parameter histograms')
    online_lme_median = np.percentile(dfPymerCoeffs_online['Time']*100.0, 50)
    # Save figure
    #outFile = '%s/Mmi-Vs-Gbe-Slopes.png'%outFigDirƒ%%
    outFile = '%s/LmeSlopeHistograms_OnlineVsApp_%s_2grp.png'%(outFigDir,batchName_app)
    print('Saving figure as %s...'%outFile)
    plt.savefig(outFile)
    print('Done!')
    online_lme_median = np.percentile(dfPymerCoeffs_online['Time']*100.0, 50)
    app_lme_median = np.percentile(dfPymerCoeffs_app['Time']*100.0, 50)
    lme_dif = online_lme_median - app_lme_median
    app_pytorch_median = np.percentile(best_pars.beta_T * 100.0, 50)
    lme_app_dif = online_lme_median - app_pytorch_median
    print(f'Online LME median slope = {online_lme_median}, app lme median = {app_lme_median}, dif = {lme_dif}')
    print(f'Online LME median slope = {online_lme_median}, app pyTorch median = {app_pytorch_median}, dif = {lme_app_dif}')

# %% Get impacts of fracRiskScore from the LME results table
batchName = 'AllOpeningRestAndRandom'
stage = 'full'
inFile = '%s/Mmi-%s_PymerFit-%s.csv'%(dataDir,batchName,stage)
print('Loading pymer fits from %s...'%inFile)
dfFits = pd.read_csv(inFile,index_col=0)
print('Done!')

m = dfFits.loc['fracRiskScore','Estimate']*100
se = dfFits.loc['fracRiskScore','SE']*100
T = dfFits.loc['fracRiskScore','T-stat']
dof = dfFits.loc['fracRiskScore', 'DF']
p = dfFits.loc['fracRiskScore','P-val']
print('Depression Risk Score x intercept in LME:')
print('%.3g +/- %.3g %% mood, T=%.3g, dof=%0.3g, p=%.3g'%(m,se,T,dof,p))

m = dfFits.loc['Time:fracRiskScore','Estimate']*100
se = dfFits.loc['Time:fracRiskScore','SE']*100
T = dfFits.loc['Time:fracRiskScore','T-stat']
dof = dfFits.loc['Time:fracRiskScore', 'DF']
p = dfFits.loc['Time:fracRiskScore','P-val']
print('Depression Risk Score x slope in LME: T=%.3g, p=%.3g'%(T,p))
print('%.3g +/- %.3g %% mood, T=%.3g, dof=%0.3g, p=%.3g'%(m,se,T,dof,p))

# %% Get mean slope in depressed and non-depressed participants

# load pymer fits
batchName = 'AllOpeningRestAndRandom'
stage = 'full'
inFile = '%s/Mmi-%s_PymerCoeffs-%s.csv'%(dataDir,batchName,stage)
print('Loading pymer fits from %s...'%inFile)
dfCoeffs = pd.read_csv(inFile)
# load fracRiskScore from same cohort
inFile = '%s/Mmi-%s_pymerInput-%s.csv'%(dataDir,batchName,stage)
print('Loading pymer input from %s...'%inFile)
dfPymerInput = pd.read_csv(inFile)
print('Done!')

participants = np.unique(dfCoeffs.Subject)
nSubj = len(participants)
fracRiskScore = np.zeros(nSubj)
slope = np.zeros(nSubj)
for i,participant in enumerate(participants):
    fracRiskScore[i] = dfPymerInput.loc[dfPymerInput.Subject==participant,'fracRiskScore'].values[0]
    slope[i] = dfCoeffs.loc[dfCoeffs.Subject==participant,'Time'].values[0]
    #ms
isAtRisk = fracRiskScore>=1
print('Mean +/- ste slope when fracRiskScore>=1: %.3f +/- %.3f'
      %(np.mean(slope[isAtRisk])*100, np.std(slope[isAtRisk]*100)/np.sqrt(np.sum(isAtRisk))))
print('Median slope when fracRiskScore>=1: %.3f'
      %(np.median(slope[isAtRisk])*100))
isNotAtRisk = fracRiskScore<1
print('Mean +/- ste slope when fracRiskScore<1: %.3f +/- %.3f'
      %(np.mean(slope[isNotAtRisk])*100, np.std(slope[isNotAtRisk]*100)/np.sqrt(np.sum(isNotAtRisk))))

# %% Depression risk vs. not
dfRating = pd.read_csv('%s/Mmi-AllOpeningRestAndRandom_pymerInput-full.csv'%(dataDir),index_col=0)
cols = dfRating.columns.tolist()
cols[cols.index('Subject')] = 'participant'
cols[cols.index('Time')] = 'time'
cols[cols.index('Mood')] = 'rating'
dfRating.columns = cols
dfRating['iBlock'] = 0
dfRating['time'] = dfRating['time']*60

participants = np.unique(dfRating.participant)
nSubj = len(participants)
lastRatingTime = np.zeros(nSubj)
firstRatingTime = np.zeros(nSubj)
nRatings = 0
for i,participant in enumerate(participants):
    firstRatingTime[i] = dfRating.loc[dfRating.participant==participant,'time'].values[0]
    lastRatingTime[i] = dfRating.loc[dfRating.participant==participant,'time'].values[nRatings-1]

# isShortSubj = lastRatingTime-firstRatingTime<410
isMediumSubj = lastRatingTime>410
isLongSubj = lastRatingTime-firstRatingTime>600
isMedium = np.isin(dfRating.participant,participants[isMediumSubj])
isLong = np.isin(dfRating.participant,participants[isLongSubj])

isAtRisk = dfRating.fracRiskScore>=1

dfTrialMean = []

# Set up figure
plt.close(511)
fig = plt.figure(511,figsize=(8,3),dpi=180, facecolor='w', edgecolor='k');
plt.clf()
# Plot results
ax1 = plt.subplot(131)
dfRatingMean0 = pmd.GetMeanRatings(dfRating.loc[~isAtRisk,:],nRatings=-1,participantLabel='Not at risk',doInterpolation=True)
dfRatingMean1 = pmd.GetMeanRatings(dfRating.loc[isAtRisk,:],nRatings=-1,participantLabel='At risk of depression',doInterpolation=True)
pmd.PlotMmiRatings(dfTrialMean,dfRatingMean0,'line',autoYlim=True, doBlockLines=False, ratingLabel=dfRatingMean0.participant[0])
pmd.PlotMmiRatings(dfTrialMean,dfRatingMean1,'line',autoYlim=True, doBlockLines=False, ratingLabel=dfRatingMean1.participant[0])
# Annotate plot
plt.axhline(0.5,c='k',ls='--',zorder=-6)#,label='neutral mood')
#plt.legend(loc='upper right')
plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.65))
plt.grid(True)
titleStr = 'Short runs \n (duration > 294 s)'
plt.title(titleStr)


# Plot results
plt.subplot(132,sharey=ax1)
dfRatingMean0 = pmd.GetMeanRatings(dfRating.loc[~isAtRisk & isMedium,:],nRatings=-1,participantLabel='Not at risk',doInterpolation=True)
dfRatingMean1 = pmd.GetMeanRatings(dfRating.loc[isAtRisk & isMedium,:],nRatings=-1,participantLabel='At risk of depression',doInterpolation=True)
pmd.PlotMmiRatings(dfTrialMean,dfRatingMean0,'line',autoYlim=True, doBlockLines=False, ratingLabel=dfRatingMean0.participant[0])
pmd.PlotMmiRatings(dfTrialMean,dfRatingMean1,'line',autoYlim=True, doBlockLines=False, ratingLabel=dfRatingMean1.participant[0])
# Annotate plot
plt.axhline(0.5,c='k',ls='--',zorder=-6)#,label='neutral mood')
#plt.legend(loc='upper right')
plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.65))
plt.grid(True)
titleStr = 'Medium runs \n (duration > 410 s)'
plt.title(titleStr)

# Plot results
plt.subplot(133,sharey=ax1)
dfRatingMean0 = pmd.GetMeanRatings(dfRating.loc[~isAtRisk & isLong,:],nRatings=-1,participantLabel='Not at risk',doInterpolation=True)
dfRatingMean1 = pmd.GetMeanRatings(dfRating.loc[isAtRisk & isLong,:],nRatings=-1,participantLabel='At risk of depression',doInterpolation=True)
pmd.PlotMmiRatings(dfTrialMean,dfRatingMean0,'line',autoYlim=True, doBlockLines=False, ratingLabel=dfRatingMean0.participant[0])
pmd.PlotMmiRatings(dfTrialMean,dfRatingMean1,'line',autoYlim=True, doBlockLines=False, ratingLabel=dfRatingMean1.participant[0])
# Annotate plot
plt.axhline(0.5,c='k',ls='--',zorder=-6)#,label='neutral mood')
#plt.legend(loc='upper right')
plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.65))
plt.grid(True)
titleStr = 'Long runs \n (duration > 600 s)'
plt.title(titleStr)
plt.ylim([0.3,0.8])

# Annotate figure
plt.tight_layout(rect=[0,0,1,0.93])
fig.subplots_adjust(bottom=0.25)
plt.suptitle('Depression risk affects mean mood ratings over time')

# Save figure
outFig = '%s/Mmi_%s_Comparison.png'%(outFigDir,'-'.join(['NotAtRisk','AtRisk']))
print('Saving figure as %s...'%outFig)
plt.savefig(outFig, bbox_inches="tight")
print('Done!')

#%%
if have_gbe:
    # %% Plot beta_T against life happiness score
    sns.set(font_scale=0.8)
    sns.set_style("whitegrid")
    alpha = 0.2
    nGrps = 2
    paramToPlot = 'beta_T'
    param = 'beta_A'

    for stage in ['full','late']:
        print('=== STAGE %s ==='%stage)
        # Load pytorch results
        if IS_EXPLORE:
            suffix = '_GbeExplore'
        else:
            suffix = '_GbeConfirm'
        if stage=='late':
            suffix = suffix + '-late'
        inFile = '%s/PyTorchParameters%s.csv'%(pytorchDir,suffix)
        print('Loading best parameters from %s...'%inFile)
        best_pars = pd.read_csv(inFile);


        if IS_EXPLORE:
            summaryFile = '%s/Mmi-GbeExplore_Summary.csv'%(dataDir)
        else:
            summaryFile = '%s/Mmi-GbeConfirm_Summary.csv'%(dataDir)
        print('Loading summary from %s..'%summaryFile)
        dfSummary = pd.read_csv(summaryFile,index_col=0)
        dfSummary['beta_T'] = best_pars['beta_T'].values
        best_pars['lifeHappy'] = dfSummary['lifeHappy'].values



        plt.close(621)
        plt.figure(621,figsize=(10,4),dpi=120)
        plt.clf()
        rs,ps = stats.spearmanr(dfSummary['lifeHappy'],dfSummary['beta_T'])
        print('lifeHappy vs. beta_T: r_s = %.3g, p_s = %.3g'%(rs,ps))

        print('Plotting lifeHappy vs. beta_T with best fit line...')
        plt.subplot(1,3,1)
        sns.regplot(x='lifeHappy', y='beta_T', data=dfSummary);
        # Annotate plot
        plt.xlabel('Life happiness rating (0-1)')
        plt.ylabel(r'$\beta_T$')
        plt.title(r'Life happiness vs. $\beta_T$:' + '\n' + r'$r_s = %.3g, p_s = %.3g$'%(rs,ps))

        plt.subplot(1,3,2)
        rs,ps = stats.spearmanr(best_pars[param],best_pars[paramToPlot])
        print('%s vs. %s: r_s = %.3g, p_s = %.3g'%(param,paramToPlot,rs,ps))

        print('Plotting %s vs. %s with best fit line...'%(param,paramToPlot))
        sns.regplot(x=param, y=paramToPlot, data=best_pars,scatter_kws={'alpha':alpha});
        plt.xlabel(paramLabelDict[param])
        plt.ylabel(paramLabelDict[paramToPlot])
        plt.title('%s vs. %s:\n'%(paramLabelDict[param],paramLabelDict[paramToPlot]) +
                                  r'$r_s = %.3g, p_s = %.3g$'%(rs,ps))

        plt.subplot(1,3,3)
        topCutoff = np.median(best_pars.lifeHappy)
        botCutoff = np.median(best_pars.lifeHappy)
        if nGrps==2:
            isTop = best_pars.lifeHappy>=topCutoff
            isBot = best_pars.lifeHappy<botCutoff
        elif nGrps==4:
            topCutoff = 0.8
            botCutoff = 0.6
        elif nGrps==11:
            topCutoff = 0.9
            botCuotff = 0.1
        nTop = np.sum(isTop)
        nBot = np.sum(isBot)
        # Run spearman corr's
        rs_top,ps_top = stats.spearmanr(best_pars.loc[isTop,param],best_pars.loc[isTop,paramToPlot])
        print('%s vs. %s (lifeHappy>=%g): r_s = %.3g, p_s = %.3g'%(param,paramToPlot,topCutoff,rs_top,ps_top))
        rs_bot,ps_bot = stats.spearmanr(best_pars.loc[isBot,param],best_pars.loc[isBot,paramToPlot])
        print('%s vs. %s (lifeHappy<%g): r_s = %.3g, p_s = %.3g'%(param,paramToPlot,botCutoff,rs_bot,ps_bot))

        # Is the diff between the two significant?
        zs_top = np.arctanh(rs_top)
        zs_bot = np.arctanh(rs_bot)
        se_diff_r = np.sqrt(1.0/(nTop - 3) + 1.0/(nBot - 3))
        diff = zs_top - zs_bot
        z = abs(diff / se_diff_r)
        p = (1 - stats.norm.cdf(z))
        #            if twotailed:
        #                p *= 2
        print('correlation difference between top & bottom: z=%.3g, p=%.3g'%(z,p))

        print('Plotting %d-group %s vs. %s with best fit lines...'%(nGrps,param,paramToPlot))
        if param=='lifeHappy':
            plt.xlim([-0.06,1.06])
        topLabel = 'Life happiness >= %g (n = %d)\n'%(topCutoff,nTop) + r'$r_s=%.3g, p_s=%.3g$'%(rs_top,ps_top)
        botLabel = 'Life happiness < %g (n = %d)\n'%(botCutoff,nBot) + r'$r_s=%.3g, p_s=%.3g$'%(rs_bot,ps_bot)
        g1 = sns.regplot(x=param, y=paramToPlot, data=best_pars.loc[isTop,:], line_kws={'color':'tab:blue','label':topLabel},scatter_kws={'color':'tab:blue','alpha':alpha});
        g2 = sns.regplot(x=param, y=paramToPlot, data=best_pars.loc[isBot,:], line_kws={'color':'tab:orange','label':botLabel},scatter_kws={'color':'tab:orange','alpha':alpha});
        plt.xlabel(paramLabelDict[param])
        plt.ylabel(paramLabelDict[paramToPlot])
        plt.title('%s vs. %s: group corr. diff.\n'%(paramLabelDict[param],paramLabelDict[paramToPlot]) +
                                  r'$z = %.3g, p = %.3g$'%(z,p))
        plt.legend()

        plt.tight_layout()
        plt.savefig('%s/PyTorch_betaT-vs-lifeHappyAndBetaA%s.png'%(outFigDir,suffix))



    # %% Plot each parameters vs. betaT
    sns.set(font_scale=0.8)
    sns.set_style("whitegrid")
    alpha = 0.2
    paramToPlot = 'beta_T'
    colsToPlot = ['m0','lambda','beta_E','beta_A','SSE','lifeHappy']
    for stage in ['full','late']:
        print('=== STAGE %s ==='%stage)
        # Load pytorch results
        if IS_EXPLORE:
            suffix = '_GbeExplore'
        else:
            suffix = '_GbeConfirm'
        if stage=='late':
            suffix = suffix + '-late'
        inFile = '%s/PyTorchParameters%s.csv'%(pytorchDir,suffix)
        print('Loading best parameters from %s...'%inFile)
        best_pars = pd.read_csv(inFile);


        if IS_EXPLORE:
            summaryFile = '%s/Mmi-GbeExplore_Summary.csv'%(dataDir)
        else:
            summaryFile = '%s/Mmi-GbeConfirm_Summary.csv'%(dataDir)
        print('Loading summary from %s..'%summaryFile)
        dfSummary = pd.read_csv(summaryFile,index_col=0)
        dfSummary['beta_T'] = best_pars['beta_T'].values
        best_pars['lifeHappy'] = dfSummary['lifeHappy'].values

        for nGrps in [1,2]:

            plt.close(621)
            plt.figure(621,figsize=(13,8),dpi=120)
            plt.clf()


            for i,param in enumerate(colsToPlot):
                plt.subplot(2,3,i+1)
                if nGrps==1:
                    rs,ps = stats.spearmanr(best_pars[param],best_pars[paramToPlot])
                    print('lifeHappy vs. beta_T: r_s = %.3g, p_s = %.3g'%(rs,ps))

                    print('Plotting %s vs. %s with best fit line...'%(param,paramToPlot))
                    rs,ps = stats.spearmanr(best_pars[param],best_pars[paramToPlot])
                    print('%s vs. %s: r_s = %.3g, p_s = %.3g'%(param,paramToPlot,rs,ps))

                    print('Plotting %s vs. %s with best fit line...'%(param,paramToPlot))
                    sns.regplot(x=param, y=paramToPlot, data=best_pars,scatter_kws={'alpha':alpha});
                    plt.xlabel(paramLabelDict[param])
                    plt.ylabel(paramLabelDict[paramToPlot])
                    plt.title('%s vs. %s:\n'%(paramLabelDict[param],paramLabelDict[paramToPlot]) +
                                              r'$r_s = %.3g, p_s = %.3g$'%(rs,ps))

                elif nGrps==2:
                    topCutoff = np.median(best_pars.lifeHappy)
                    botCutoff = np.median(best_pars.lifeHappy)
                    if nGrps==2:
                        isTop = best_pars.lifeHappy>=topCutoff
                        isBot = best_pars.lifeHappy<botCutoff
                    elif nGrps==4:
                        topCutoff = 0.8
                        botCutoff = 0.6
                    elif nGrps==11:
                        topCutoff = 0.9
                        botCuotff = 0.1
                    nTop = np.sum(isTop)
                    nBot = np.sum(isBot)
                    # Run spearman corr's
                    rs_top,ps_top = stats.spearmanr(best_pars.loc[isTop,param],best_pars.loc[isTop,paramToPlot])
                    print('%s vs. %s (lifeHappy>=%g): r_s = %.3g, p_s = %.3g'%(param,paramToPlot,topCutoff,rs_top,ps_top))
                    rs_bot,ps_bot = stats.spearmanr(best_pars.loc[isBot,param],best_pars.loc[isBot,paramToPlot])
                    print('%s vs. %s (lifeHappy<%g): r_s = %.3g, p_s = %.3g'%(param,paramToPlot,botCutoff,rs_bot,ps_bot))

                    # Is the diff between the two significant?
                    zs_top = np.arctanh(rs_top)
                    zs_bot = np.arctanh(rs_bot)
                    se_diff_r = np.sqrt(1.0/(nTop - 3) + 1.0/(nBot - 3))
                    diff = zs_top - zs_bot
                    z = abs(diff / se_diff_r)
                    p = (1 - stats.norm.cdf(z))
                    #            if twotailed:
                    #                p *= 2
                    print('correlation difference between top & bottom: z=%.3g, p=%.3g'%(z,p))

                    print('Plotting %d-group %s vs. %s with best fit lines...'%(nGrps,param,paramToPlot))
                    if param=='lifeHappy':
                        plt.xlim([-0.06,1.06])
                    topLabel = 'Life happiness >= %g (n = %d)\n'%(topCutoff,nTop) + r'$r_s=%.3g, p_s=%.3g$'%(rs_top,ps_top)
                    botLabel = 'Life happiness < %g (n = %d)\n'%(botCutoff,nBot) + r'$r_s=%.3g, p_s=%.3g$'%(rs_bot,ps_bot)
                    g1 = sns.regplot(x=param, y=paramToPlot, data=best_pars.loc[isTop,:], line_kws={'color':'tab:blue','label':topLabel},scatter_kws={'color':'tab:blue','alpha':alpha});
                    g2 = sns.regplot(x=param, y=paramToPlot, data=best_pars.loc[isBot,:], line_kws={'color':'tab:orange','label':botLabel},scatter_kws={'color':'tab:orange','alpha':alpha});
                    plt.xlabel(paramLabelDict[param])
                    plt.ylabel(paramLabelDict[paramToPlot])
                    plt.title('%s vs. %s: group corr. diff.\n'%(paramLabelDict[param],paramLabelDict[paramToPlot]) +
                                              r'$z = %.3g, p = %.3g$'%(z,p))
                    plt.legend()

                plt.tight_layout()




            plt.tight_layout()
            plt.savefig('%s/PyTorch_betaT-vs-others%s-%dGrps.png'%(outFigDir,suffix,nGrps))


# %% Get impacts of fracRiskScore from the LME results table
batchName = 'AllOpeningRestAndRandom'
stage = 'full'
inFile = '%s/Mmi-%s_PymerFit-%s.csv'%(dataDir,batchName,stage)
print('Loading pymer fits from %s...'%inFile)
dfFits = pd.read_csv(inFile,index_col=0)
print('Done!')

m = dfFits.loc['isAge16to18TRUE','Estimate']*100
se = dfFits.loc['isAge16to18TRUE','SE']*100
T = dfFits.loc['isAge16to18TRUE','T-stat']
dof = dfFits.loc['isAge16to18TRUE','DF']
p = dfFits.loc['isAge16to18TRUE','P-val']
print('Age 16-18 x intercept in LME:')
print('%.3g +/- %.3g %% mood, T=%.3g, dof=%0.3g, p=%.3g'%(m,se,T,dof,p))

m = dfFits.loc['Time:isAge16to18TRUE','Estimate']*100
se = dfFits.loc['Time:isAge16to18TRUE','SE']*100
T = dfFits.loc['Time:isAge16to18TRUE','T-stat']
dof = dfFits.loc['Time:isAge16to18TRUE','DF']
p = dfFits.loc['Time:isAge16to18TRUE','P-val']
print('Age 16-18 x slope in LME:')
print('%.3g +/- %.3g %% mood, T=%.3g, dof=%0.3g, p=%.3g'%(m,se,T,dof,p))

# %% Link to age in adolescents
PlotAgeVsCoeffs('AllOpeningRestAndRandom')

# %% Get Stability plots

# Set up
plt.close(923)
plt.figure(923,figsize=(9,6),dpi=120); plt.clf()
intOrSlopes = ['Intercept','Slope']
cohortPairs = [['Stability01-Rest','Stability01-Rest_block2'],
               ['Stability01-Rest','Stability02-Rest'],
               ['COVID01','COVID03']]
pairTitles = ['Blocks','Days','Weeks']
# Calculate and plot ICCs
icc21 = {'Intercept':0,'Slope':0}
p21 = {'Intercept':0,'Slope':0}
for i,pair in enumerate(cohortPairs):
    icc21['Intercept'],p21['Intercept'],icc21['Slope'],p21['Slope'] = gmi.GetMmiIcc(pair[0],pair[1],doPlot='None')
    for j,intOrSlope in enumerate(intOrSlopes):
        ax = plt.subplot(2,3,j*3+i+1);
        gmi.PlotReliability(pair[0],pair[1],intOrSlope=intOrSlope)
        if j==0:
            plt.title('%s\nICC(2,1)=%.3g, p=%.3g'%(pairTitles[i],icc21[intOrSlope],p21[intOrSlope]))
        else:
            plt.title('ICC(2,1)=%.3g, p=%.3g'%(icc21[intOrSlope],p21[intOrSlope]))

# Save figure
plt.tight_layout()
outFile = '%s/Mmi_%s_Reliability.png'%(outFigDir,'-'.join(pairTitles))
print('Saving figure as %s...'%outFile)
plt.savefig(outFile)
print('Done!')


# %% Check for time of day effects
PlotTimeOfDayVsSlopeAndIntercept('AllOpeningRestAndRandom')

# %% Impact of mood on gambling

def CompareGamblingBehavior(dataDir,outFigDir,batchNames,groupName,batchLabels,iGambleBlock,nGamble=4):
    minNRatings=8 # -1 indicates all, but they must be the same
    minNTrials=10 # -1 indicates all, but they must be the same
    xlim=[0,90]
    bar_ylim=[0.6,0.92]
    nChoseGamble = [0]* len(batchNames)
    participants = [0]* len(batchNames)
    #plt.rcParams.update({'font.size': 6})
    plt.close(412)
    plt.figure(412,figsize=(6,7.5),dpi=180, facecolor='w', edgecolor='k');
    plt.clf();
    fig, ax = plt.subplots(3,1,num=412)
    meanGamble = np.zeros(len(batchNames))
    steGamble = np.zeros(len(batchNames))
    ratingLabels = list(batchLabels)
    # Get gambling behavior for each
    for iBatch, batchName in enumerate(batchNames):
        dfRating = pd.read_csv('%s/Mmi-%s_Ratings.csv'%(dataDir,batchName))
        dfTrial = pd.read_csv('%s/Mmi-%s_Trial.csv'%(dataDir,batchName))

        # Limit to block
        iBlock = iGambleBlock[iBatch]
        if iBlock!='all':
            dfRating = dfRating.loc[dfRating.iBlock==iBlock,:]
            dfTrial = dfTrial.loc[dfTrial.iBlock==iBlock,:]
        # Get averages
        dfRatingMean = pmd.GetMeanRatings(dfRating,minNRatings)
        dfTrialMean = pmd.GetMeanTrials(dfTrial,minNTrials)
        participants[iBatch] = np.unique(dfTrial.participant)
        nSubj = len(participants[iBatch])

        # Get average gambleFrac in first nGamble trials for each subject
        nChoseGamble[iBatch] = np.zeros(nSubj)
        for iSubj,subj in enumerate(participants[iBatch]):
            isGamble = dfTrial.loc[dfTrial.participant==subj].values =='gamble'
            nChoseGamble[iBatch][iSubj] = np.sum(isGamble[:nGamble])

        # Plot results
        ratingLabels[iBatch] = '%s (n = %d)'%(batchLabels[iBatch],nSubj)
    #    print(dfTrialMean.shape[0])

        plt.sca(ax[0]) #plt.subplot(311)
        dfTrialMean.time = dfTrialMean.time - dfTrialMean.time[0]
        dfRatingMean.time = dfRatingMean.time - dfRatingMean.time[0]

        pmd.PlotMmiRatings(dfTrialMean,dfRatingMean,interp='linear',autoYlim=True,doBlockLines=False,ratingLabel=ratingLabels[iBatch])
    #    plt.plot(dfRatingMean.time - dfRatingMean.time[0],dfRatingMean.rating,'.-',label=ratingLabels[iBatch])
        plt.xlabel('Time from block start (s)')
        plt.ylabel('Mood (0-1)')
        plt.legend(loc=4)
        plt.title('')
        plt.xlim(xlim)
        plt.grid(True,zorder=-3)

        # plot fraction gambling on each trial
        plt.sca(ax[1]) #plt.subplot(312)
        plt.plot(dfTrialMean.time,dfTrialMean.gambleFrac,'.-',label=batchLabels[iBatch])
        # add 95% confidence interval patch
        plt.fill_between(dfTrialMean.time,dfTrialMean.gambleFracCIMin,
                                 dfTrialMean.gambleFracCIMax,alpha=0.5,zorder=0)
        # annotate plot
        plt.xlabel('Time from block start (s)')
        plt.ylabel('Fraction choosing to gamble')
        plt.legend(loc=4)
        plt.xlim(xlim)
        plt.grid(True,zorder=-3)

        plt.sca(ax[2]) # plt.subplot(313)
        meanGamble[iBatch] = np.mean(dfTrialMean.gambleFrac[:nGamble])
        plt.bar(iBatch,meanGamble[iBatch],zorder=2)
        steGamble[iBatch] = np.std(1.0*nChoseGamble[iBatch]/nGamble)/np.sqrt(nSubj)
        plt.plot([iBatch,iBatch],[meanGamble[iBatch]-steGamble[iBatch],
                  meanGamble[iBatch]+steGamble[iBatch]],'k-')
        plt.ylabel('Fraction choosing to gamble\nin first %d trials'%nGamble)
        plt.ylim(bar_ylim)
        plt.grid(True,zorder=-3)


    # Add stars for stats tests
    nGrps = len(batchNames)
    p=np.zeros((nGrps,nGrps))
    stat=np.zeros((nGrps,nGrps))
    dof=np.zeros((nGrps,nGrps))
    print('=== Ranksum tests on subject-wise gamble proportions: ===')
    nComparisons = nGrps*(nGrps-1)/2;
    cutoff = 0.05/nComparisons;
    isAnyStar = False
    for i in range(nGrps-1):
        for j in range(i+1,nGrps):
            stat[i,j] ,p[i,j] = stats.ranksums(nChoseGamble[i],nChoseGamble[j])
            p[j,i] = p[i,j]
            stat[j,i] = stat[i,j]
            dof[j,i] = dof[i,j] = len(nChoseGamble[i]) + len(nChoseGamble[j]) - 2
            if p[i,j]<cutoff:
                yMax = bar_ylim[1]
                yStars = [yMax-0.07+(i+j)*.04, yMax-0.05+(i+j)*.04, yMax-0.05+(i+j)*.04, yMax-0.07+(i+j)*.04]
                plt.plot([i,i,j,j],yStars ,'k-')
                if isAnyStar:
                    plt.plot((i+j)/2.0,yMax-0.03+(i+j)*.04,'k*')
                else:
                    plt.plot((i+j)/2.0,yMax-0.03+(i+j)*.04,'k*',label='p < 0.05/%d'%nComparisons)
                    isAnyStar = True
            print('%s vs. %s: stat=%.3g, dof=%.3g, p=%.3g'%(batchLabels[i],batchLabels[j],stat[i,j],dof[i,j], p[i,j]))

    #plt.legend(loc='lower left')
    plt.xticks(np.arange(len(batchNames)),batchLabels)
    plt.ylim([0.6,1])
    plt.tight_layout(rect=[0,0,1,0.95])
    figTitle='Opening rest period is associated with reduced gambling choices'
    plt.suptitle('%s'%figTitle)
    outFile = '%s/RestDurationComparison_%s.png'%(outFigDir,groupName)
    print('Saving figure as %s...'%outFile)
    plt.savefig(outFile)
    print('Done!')

    return participants,nChoseGamble

# Compare no opening rest, short opening rest, and long opening rest
batchNames = ['NoOpeningRest','ShortOpeningRest','LongOpeningRest'];
groupName = 'No-Short-Long'
batchLabels = ['No rest','350-450 s rest','500-700 s rest']
iGambleBlock = [0,1,1] # which was first gambling block in each batch
nGamble = 4 # number of initial trials to average gambleFrac in
CompareGamblingBehavior(dataDir,outFigDir,batchNames,groupName,batchLabels,iGambleBlock,nGamble=4)

# Compare no opening rest and any opening rest
batchNames = ['NoOpeningRest','AnyOpeningRest'];
groupName = 'No-Any'
batchLabels = ['No rest','Any rest']
iGambleBlock = [0,1] # which was first gambling block in each batch
nGamble = 4 # number of initial trials to average gambleFrac in
participants, nChoseGamble = CompareGamblingBehavior(dataDir,outFigDir,batchNames,groupName,batchLabels,iGambleBlock,nGamble=4)

# Compare
participants = participants[1]
nChoseGamble = nChoseGamble[1]
batchName = 'AllOpeningRestAndRandom' # should include any participants in AnyOpeningRest
stage = 'full'
inFile = '%s/Mmi-%s_PymerCoeffs-%s.csv'%(dataDir,batchName,stage)
print('Loading pymer fits from %s...'%inFile)
dfCoeffs = pd.read_csv(inFile)
# get slope for each participant
coeffs = np.zeros_like(nChoseGamble)
for iSubj,subj in enumerate(participants):
    print(subj)
    coeffs[iSubj] = dfCoeffs.loc[np.abs(dfCoeffs.Subject)==subj,'Time'].values[0] # abs because we made NIMH participant numbers negative
# do stats test
rs,ps = stats.spearmanr(coeffs,nChoseGamble)
print(f'{batchName} nChoseGamble vs. LME mood slope: r_s={rs:.3g}, p_s={ps:.3g}')

# %% Plot m0 against life happiness score

# load pymer fits
batchName = 'AllOpeningRestAndRandom'
stage = 'full'
inFile = '%s/Mmi-%s_PymerCoeffs-%s.csv'%(dataDir,batchName,stage)
print('Loading pymer fits from %s...'%inFile)
dfCoeffs = pd.read_csv(inFile)
# load fracRiskScore from same cohort
inFile = '%s/Mmi-%s_pymerInput-%s.csv'%(dataDir,batchName,stage)
print('Loading pymer input from %s...'%inFile)
dfPymerInput = pd.read_csv(inFile)
print('Done!')
# load life happiness from same cohort
inFile = '%s/Mmi-%s_LifeHappy.csv'%(dataDir,batchName)
print('Loading life happiness ratings from %s...'%inFile)
dfLifeHappy = pd.read_csv(inFile)


for i in range(dfCoeffs.shape[0]):
    subj = dfCoeffs.loc[i,'Subject']
    try:
        dfCoeffs.loc[i,'lifeHappy'] = dfLifeHappy.loc[dfLifeHappy.participant==subj,'rating'].values
        dfCoeffs.loc[i,'fracRiskScore'] = dfPymerInput.loc[dfPymerInput.Subject==subj,'fracRiskScore'].values[0]
    except:
        print('subj %d not found!'%subj)

# Load comp model params
if IS_EXPLORE:
    suffix = '_GbeExplore'
else:
    suffix = '_GbeConfirm'
for stage in ['full','late']:
    if stage=='late':
        suffix = suffix + '-late'

    if have_gbe:
        inFile = '%s/PyTorchParameters%s.csv'%(pytorchDir,suffix)
        print('Loading pyTorch parameters from %s...'%inFile)
        dfParams = pd.read_csv(inFile,index_col=0)
        # load life happiness from same cohort
        if IS_EXPLORE:
            dfSummary = pd.read_csv('%s/Mmi-GbeExplore_Summary.csv'%dataDir)
        else:
            dfSummary = pd.read_csv('%s/Mmi-GbeConfirm_Summary.csv'%dataDir)

    plt.close(841)
    doRisk = False
    if doRisk:
        plt.figure(841,figsize=[12,4],dpi=120)
        plt.subplot(1,3,1)
        plt.scatter(dfCoeffs['(Intercept)'],dfCoeffs['fracRiskScore'])
        rs,ps = stats.spearmanr(dfCoeffs['(Intercept)'],dfCoeffs['fracRiskScore'])
        plt.grid(True)
        plt.xlabel(r'Initial mood parameter $M_0$')
        plt.ylabel('Depression risk score')
        plt.title(r'Online cohort LME' + '\n' + r'$r_s=%.3g$, $p_s=%.3g$'%(rs,ps))

        plt.subplot(1,3,2)
    else:
        plt.figure(841,figsize=[9,4],dpi=120)
        plt.subplot(1,2,1)
    # plot LME intercept vs. lifeHappy
    plt.scatter(dfCoeffs['(Intercept)'],dfCoeffs['lifeHappy'],alpha=0.5)
    rs,ps = stats.spearmanr(dfCoeffs['(Intercept)'],dfCoeffs['lifeHappy'])
    plt.grid(True)
    plt.xlabel(r'Initial mood parameter $M_0$')
    plt.ylabel('Life happiness')
    plt.title(r'Online cohort LME' + '\n' + r'$r_s=%.3g$, $p_s=%.3g$'%(rs,ps))

    # plot LTA M0 parameter vs. lifeHappy
    if have_gbe:
        if doRisk:
            plt.subplot(1,3,3)
        else:
            plt.subplot(1,2,2)

        plt.scatter(dfParams['m0'],dfSummary['lifeHappy']+np.random.rand(dfSummary.shape[0])*0.05,alpha=0.01)
        #plt.hist2d(dfParams['m0'],dfSummary['lifeHappy'], bins=(np.arange(0,1.04,0.02)-0.01, np.arange(0,1.2,0.1)-0.05), cmap=plt.get_cmap('Blues'))
        rs,ps = stats.spearmanr(dfParams['m0'],dfSummary['lifeHappy'])
        plt.grid(True)
        plt.xlabel(r'Initial mood parameter $M_0$')
        plt.ylabel('Life happiness')
        plt.title(r'Mobile app cohort computational model' + '\n' + r'$r_s=%.3g$, $p_s=%.3g$'%(rs,ps))

    plt.tight_layout()
    outFile = '%s/PyTorchAndLme_m0-vs-lifeHappy%s.png'%(outFigDir,suffix)
    print('Saving figure as %s...'%outFile)
    plt.savefig(outFile)
    print('Done!')


# %% Plot LME slopes of app and online cohorts

batchName_online = 'AllOpeningRestAndRandom'
if IS_EXPLORE:
    batchName_app = 'GbeExplore'
else:
    batchName_app = 'GbeConfirm'
#dfPymerFit = pd.read_csv('%s/Mmi-%s_pymerFit-full.csv'%(dataDir,batchName),index_col=0)
dfPymerCoeffs_online = pd.read_csv('%s/Mmi-%s_pymerCoeffs-full.csv'%(dataDir,batchName_online),index_col=0)
if have_gbe:
    dfPymerCoeffs_app = pd.read_csv('%s/Mmi-%s_pymerCoeffs-full.csv'%(dataDir,batchName_app),index_col=0)

# Load pytorch results
if IS_EXPLORE:
    suffix = '_GbeExplore'
else:
    suffix = '_GbeConfirm'

for stage in ['full','late']:
    if stage=='late':
        suffix = suffix + '-late'

    if have_gbe:
        inFile = '%s/PyTorchParameters%s.csv'%(pytorchDir,suffix)
        print('Loading best parameters from %s...'%inFile)
        best_pars = pd.read_csv(inFile);

    # Set up Plot
    plt.close(623)
    plt.figure(623,figsize=[10,4],dpi=200)
    plt.clf()

    # Plot histograms
    xHist = np.linspace(-10.0,10.0,100)
    nSubj_online = dfPymerCoeffs_online.shape[0]
    weights = np.ones(nSubj_online)/nSubj_online*100
    plt.hist(dfPymerCoeffs_online['Time']*100.0,xHist,weights=weights,alpha=0.5,label='All online participants (n=%d), LME'%nSubj_online)
    if have_gbe:
        nSubj_app = dfPymerCoeffs_app.shape[0]
        weights = np.ones(nSubj_app)/nSubj_app*100
        if IS_EXPLORE:
            plt.hist(dfPymerCoeffs_app['Time']*100.0,xHist,weights=weights,alpha=0.5,label='Exploratory mobile app participants (n=%d), LME'%nSubj_app)
        else:
            plt.hist(dfPymerCoeffs_app['Time']*100.0,xHist,weights=weights,alpha=0.5,label='Confirmatory mobile app participants (n=%d), LME'%nSubj_app)

        nSubj_model = best_pars.shape[0]
        weights = np.ones_like(best_pars.beta_T) * 100.0 / nSubj_model
        if IS_EXPLORE:
            plt.hist(best_pars.beta_T*100,xHist,weights=weights,alpha=0.6,zorder=3,label='Exploratory mobile app participants (n=%d), computational model'%nSubj_model)
        else:
            plt.hist(best_pars.beta_T*100,xHist,weights=weights,alpha=0.6,zorder=3,label='Confirmatory mobile app participants (n=%d), computational model'%nSubj_model)

    # Annotate plot
    plt.grid(True)
    plt.xlabel('LME slope parameter or computational\nmodel time sensitivity parameter (% mood/min)')
    plt.ylabel('Percent of participants')
    plt.legend()
    plt.title('Mood slope parameter distributions vary with analysis choice')
    plt.tight_layout()
    # Save figure
    outFile = '%s/LmeSlopeHistograms_OnlineVsApp%s.png'%(outFigDir,suffix)
    print('Saving figure as %s...'%outFile)
    plt.savefig(outFile)
    print('Done!')

# %% Check for stats differences between above cohorts/analyses

if have_gbe:
    batchName_online = 'AllOpeningRestAndRandom'
    if IS_EXPLORE:
        batchName_app = 'GbeExplore'
    else:
        batchName_app = 'GbeConfirm'

    for stage in ['full','late']:
        print('=== STAGE %s ==='%stage)
        #dfPymerFit = pd.read_csv('%s/Mmi-%s_pymerFit-full.csv'%(dataDir,batchName),index_col=0)
        dfPymerCoeffs_online = pd.read_csv('%s/Mmi-%s_pymerCoeffs-%s.csv'%(dataDir,batchName_online,stage),index_col=0)
        dfPymerCoeffs_app = pd.read_csv('%s/Mmi-%s_pymerCoeffs-%s.csv'%(dataDir,batchName_app,stage),index_col=0)

        # Load pytorch results
        if IS_EXPLORE:
            suffix = '_GbeExplore'
        else:
            suffix = '_GbeConfirm'
        if stage=='late':
            suffix = suffix + '-late'
        inFile = '%s/PyTorchParameters%s.csv'%(pytorchDir,suffix)
        print('Loading best parameters from %s...'%inFile)
        best_pars = pd.read_csv(inFile);

        print('median pymer online slope: %.3g'%np.median(dfPymerCoeffs_online['Time']))
        print('median pymer app slope: %.3g'%np.median(dfPymerCoeffs_app['Time']))
        print('median comp. model app slope: %.3g'%np.median(best_pars['beta_T']))
        stat,p = stats.ranksums(dfPymerCoeffs_online['Time'],dfPymerCoeffs_app['Time'])
        dof = len(dfPymerCoeffs_online['Time']) + len(dfPymerCoeffs_app['Time']) - 2
        print(f'pymer online vs. pymer app: ranksum stat = {stat:.3g}, dof={dof}, p={p:0.3g}')

        stat,p = stats.ranksums(dfPymerCoeffs_online['Time'],best_pars['beta_T'])
        dof = len(dfPymerCoeffs_online['Time']) + len(best_pars['beta_T']) - 2
        print(f'pymer online vs. comp. model app: ranksum stat = {stat:.3g}, dof={dof}, p={p:0.3g}')
        
        stat,p = stats.ranksums(dfPymerCoeffs_app['Time'],best_pars['beta_T'])
        dof = len(dfPymerCoeffs_app['Time']) + len(best_pars['beta_T']) - 2
        print(f'pymer app vs. comp. model app: ranksum stat = {stat:.3g}, dof={dof}, p={p:0.3g}')

        print('pymer online: %.1f%% participants had beta_T<0'%(np.mean(dfPymerCoeffs_online['Time']<0)*100))
        print('pymer app: %.1f%% participants had beta_T<0'%(np.mean(dfPymerCoeffs_app['Time']<0)*100))
        print('comp model app: %.1f%% participants had beta_T<0'%(np.mean(best_pars['beta_T']<0)*100))



# %% Print LME table in LaTeX format
batchName = 'AllOpeningRestAndRandom'
dfPymerFit = pd.read_csv('%s/Mmi-%s_pymerFit-full.csv'%(dataDir,batchName),index_col=0)
print(f'{batchName} LME fit table')
print(dfPymerFit.to_latex(float_format='%.3g'))

# %% Check whether amplitude of residuals are equal over time

if have_gbe:
    if IS_EXPLORE:
        suffix = '_GbeExplore'
    else:
        suffix = '_GbeConfirm'
    # Load results
    inFile = '%s/PyTorchPredictions%s.npy'%(pytorchDir,suffix)
    print('Loading pyTorch best fits from %s...'%inFile)
    fits = np.load(inFile)
    n_trials,n_subjects = fits.shape
    print('Done!')

    # Load data
    if IS_EXPLORE:
        cohort = 'gbeExplore'
        inFile = '%s/Mmi-%s_TrialForMdls.csv'%(pytorchDir,cohort)
    else:
        cohort = 'GbeConfirm'
        inFile = '%s/Mmi-%s_TrialForMdls.csv'%(pytorchDir,cohort)
    print('Loading actual mood data from %s...'%inFile)
    dfData = pd.read_csv(inFile,index_col=0)
    mood = dfData['happySlider.response'].values.reshape((-1,n_trials)).T
    tMood = dfData['time'].values.reshape((-1,n_trials)).T

    residuals = fits-mood
    meanResiduals = np.mean(residuals,axis=1)
    rmsResiduals = np.sqrt(np.mean(residuals**2,axis=1))
    isRating = ~np.isnan(rmsResiduals)

    # Set up Plot
    plt.close(689)
    plt.figure(689,figsize=[10,4],dpi=200)
    plt.clf()
    plt.plot(meanResiduals[isRating],'.-',label='mean')
    plt.plot(rmsResiduals[isRating],'.-',label='RMS')
    plt.grid(True)
    plt.legend()
    plt.xlabel('Rating')
    plt.ylabel('Residuals of computational model (fit-mood)')
    plt.title('Pytorch model fits over ratings on %s cohort'%cohort)
    plt.tight_layout()
    # Save figure
    outFile = '%s/PytorchResidualsVsTime_%s%s.png'%(outFigDir,cohort,suffix)
    print('Saving figure as %s...'%outFile)
    plt.savefig(outFile)
    print('Done!')

# %% Test whether slope is correlated with number of plays
# Load results

if have_gbe:
    if IS_EXPLORE:
        suffix = '_GbeExplore'
        cohort = 'Exploratory Mobile App'
    else:
        suffix = '_GbeConfirm'
        cohort = 'Confirmatory Mobile App'

    for stage in ['full','late']:
        if stage=='late':
            suffix = suffix + '-late'

        inFile = '%s/PyTorchParameters%s.csv'%(pytorchDir,suffix)
        print('Loading pyTorch best fits from %s...'%inFile)
        dfParams = pd.read_csv(inFile,index_col=0)

        # get number of plays
        inFile = '%s/Mmi-GbeConfirm_Summary.csv'%dataDir
        print('Loading summary data from %s...'%inFile)
        dfSummary = pd.read_csv(inFile,index_col=0)
        print('Adding number of plays...')
        for subj in dfParams.participant:
            dfParams.loc[dfParams.participant==subj,'noPlays'] = dfSummary.loc[dfSummary.participant==subj,'noPlays'].values

        # Correlate
        print('beta_T vs. number of plays:')
        rs,ps = stats.spearmanr(dfParams.beta_T,dfParams.noPlays)
        print('rs=%.3g, ps=%.3g'%(rs,ps))
        stat,p = stats.ranksums(dfParams.loc[dfParams.noPlays==1,'beta_T'],
                             dfParams.loc[dfParams.noPlays>1,'beta_T'])
        dof = len(dfParams.loc[dfParams.noPlays==1,'beta_T']) + len(dfParams.loc[dfParams.noPlays>1,'beta_T']) - 2
        print(f'ranksum: stat = {stat:.3g}, dof = {dof}, p={p:.3g}')

        plt.figure(692);
        plt.clf();
        # get betaTs in % mood/min
        xHist = np.linspace(np.min(dfParams.beta_T),np.max(dfParams.beta_T),30)*100
        y2 = dfParams.loc[dfParams.noPlays==1,'beta_T']*100
        y1 = dfParams.loc[dfParams.noPlays>1,'beta_T']*100
        weights = np.ones_like(y1) * 100.0 / len(y1)
        plt.hist(y1,xHist,weights=weights,alpha=0.4,label='Played again (n=%d)'%len(y1))
        weights = np.ones_like(y2) * 100.0 / len(y2)
        plt.hist(y2,xHist,weights=weights,alpha=0.4,label='Did not play again (n=%d)'%len(y2))
        # annotate plot
        plt.legend()
        plt.grid(True)
        plt.xlabel(r'Time sensitivity parameter $\beta_T$ (% mood/min)')
        plt.ylabel('Percent of participants')
        plt.title('Time sensitivity vs. choice to play again, %s cohort'%cohort)
        plt.tight_layout()
        # Save figure
        outFile = '%s/PytorchBetaT-Vs-NoPlays%s.png'%(outFigDir,suffix)
        print('Saving figure as %s...'%outFile)
        plt.savefig(outFile)
        print('Done!')

# %% Check whether initial mood rating correlates

if have_gbe:
    # Load data
    if IS_EXPLORE:
        cohort = 'gbeExplore'
    else:
        cohort = 'GbeConfirm'
    # load mood rating
    inFile = '%s/Mmi-%s_Ratings.csv'%(dataDir,cohort)
    print('Loading ratings data from %s...'%inFile)
    dfRatings = pd.read_csv(inFile,index_col=0)
    nSubj = len(np.unique(dfRatings.participant))
    initMood = dfRatings['rating'].values.reshape((nSubj,-1))[:,0]
    # load trial data
    inFile = '%s/Mmi-%s_Trial.csv'%(dataDir,cohort)
    print('Loading trial data from %s...'%inFile)
    dfTrial = pd.read_csv(inFile)

    nGamble = 4 # trials on which to assess gambling

    # Get average gambleFrac in first nGamble trials for each subject
    nChoseGamble = np.zeros(nSubj)
    for iSubj,subj in enumerate(np.unique(dfTrial.participant)):
        isGamble = dfTrial.loc[dfTrial.participant==subj].values =='gamble'
        nChoseGamble[iSubj] = np.sum(isGamble[:nGamble])
    # print Spearman correlation
    rs,ps = stats.spearmanr(initMood,nChoseGamble)
    print('%s first mood rating vs. gambling in 1st %d trials:'%(cohort,nGamble))
    print('rs = %.3g, ps = %.3g'%(rs,ps))


# %% Make joint plot of LME initial mood and slopes
# load pymer fits
batchName = 'AllOpeningRestAndRandom'
stage = 'full'
inFile = '%s/Mmi-%s_PymerCoeffs-%s.csv'%(dataDir,batchName,stage)
print('Loading pymer fits from %s...'%inFile)
dfCoeffs = pd.read_csv(inFile)
# Make joint plot
PlotPymerHistosJoint(dfCoeffs)
nSubj = dfCoeffs.shape[0]
plt.suptitle('LME Parameters for all subjects with opening \nrest or random gambling (n=%d)'%nSubj)
# Save resulting figure
outFile = '%s/PymerCoeffJointPlot_%s-%s.png'%(outFigDir,batchName,stage)
print('Saving figure as %s...'%outFile)
plt.savefig(outFile)
print('Done!')

# %% Test for multitasking
batchName = 'AllOpeningRestAndRandom'
# load ratings
dfRating = pd.read_csv('%s/Mmi-%s_Ratings.csv'%(dataDir,batchName), index_col=0)
# check which trials were locked in or moved.
isLockedIn = pd.notna(dfRating.RT).values
isMoved = dfRating.rating!=0.5
pctLockedOrMoved = np.mean(isLockedIn | isMoved) * 100
print(f'Batch {batchName}: Mood ratings were locked in or moved on {pctLockedOrMoved}% of trials.')
