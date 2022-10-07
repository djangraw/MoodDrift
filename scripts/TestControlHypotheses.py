#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TestControlsHypotheses.py

Test the control analyses preregistered on osf.

Created on Thu Dec 16 15:27:29 2021
@author: djangraw
- Updated 12/22/21 by DJ - finished script, commented.
- Updated 1/5/22 by DJ - added cohen's d calculations, effect size reports,
   and activities summary
- Updated 8-9/22 by DN & DJ - switch to within- and between-subject R2 values, many updates
"""

# Import packages
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import seaborn as sns
# Import pymer functions
from pymer4.models import Lmer
from rpy2 import robjects
mumin = robjects.r('library(MuMIn)') # for F-test

# %% Declare functions
# LMER comparison function from Dylan
def compare_lmers(pymer_input, lm_string_a, lm_string_ab):

    """
    Run an anova comparing to mixed effects models on the same data.
    Note that the mixed effects structrure for both models should be the same.
    Also calculate r-squared and f-squared based on marginal r-squared.

    Parameters
    ----------
   pymer_input : Pandas.DataFrame
        Data to run the models on
    lm_string_a : str
        lme4 model string for the reduced model
    lm_string_ab : str
        lme4 model string for the more complex model

    Returns
    -------
    anova_res : Pandas.DataFrame
        Results of the anova including marginal and conditional r-squared and f-squared.

    References
    ----------
    Formula for f-squared:
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3328081/#:~:text=Cohen's%20f%202%20(Cohen%2C%201988,R%202%201%20%2D%20R%202%20.

    Where I found the r-squared for mixed effects models function:
    https://stats.stackexchange.com/a/347908
    """
    # fit more complex LME model to to the data
    print('*** Fitting more complex model... ***')
    model_ab = Lmer(lm_string_ab, data=pymer_input)
    _ = model_ab.fit(REML=False, old_optimizer=True)
    dfFit_ab = model_ab.coefs
    dfFixef_ab = model_ab.fixef
    # fit reduced complex LME model to to the data
    print('*** Fitting reduced model... ***')
    model_a = Lmer(lm_string_a, data=pymer_input)
    _ = model_a.fit(REML=False, old_optimizer=True)
    dfFit_a = model_a.coefs
    dfFixef_a = model_a.fixef
    # run anova to compare variance explained
    anova_res = pd.DataFrame((robjects.r('anova')(model_a.model_obj, model_ab.model_obj, refit=False)))
    # double-check that reduced model is actually smaller
    if len(dfFit_ab) <= (len(dfFit_a)):
        raise ValueError("Model AB should be a larger model than Model A, but based on the fixed effects, that's not the case.")
    # get rsquared values
    rsquaredab = robjects.r('r.squaredGLMM')(model_ab.model_obj)[0]
    rsquareda = robjects.r('r.squaredGLMM')(model_a.model_obj)[0]
    r2lr = robjects.r('''
    function (object)
    {
        r.squaredLR(object, null.RE = TRUE, adj.r.squared=TRUE)
    }
    ''')
    rsquaredab_lr = r2lr(model_ab.model_obj)[0]
    rsquareda_lr = r2lr(model_a.model_obj)[0]
    # add rsquared and F-test results to anova table
    anova_res['marginal_R2'] = (rsquareda[0], rsquaredab[0])
    anova_res['conditional_R2'] = (rsquareda[1], rsquaredab[1])
    anova_res['lr_marginal_R2'] = (rsquareda_lr, rsquaredab_lr)

    anova_res['f2'] = (np.nan, (rsquaredab[0] - rsquareda[0]) / (1 - (rsquaredab[0])))
    anova_res['lr_f2'] = (np.nan, (rsquaredab_lr - rsquareda_lr) / (1 - (rsquaredab_lr)))


    #dfFit_a['VIF'] = np.insert(robjects.r('vif')(model_a.model_obj),0,0)
    #dfFit_ab['VIF'] = np.insert(robjects.r('vif')(model_ab.model_obj),0,0)
    # return results
    return anova_res, dfFit_a, dfFit_ab, dfFixef_a

# Function to get summary boredom scores for each participant
def GetBoredomScores(df_boredom_probes):
    # Extract unique list of participant & blocks
    participants = np.unique(df_boredom_probes.participant)
    blocks = np.unique(df_boredom_probes.iBlock).astype(int)
    # Set up summary table
    df_summary = pd.DataFrame(columns=['participant']+[f'block{x}' for x in blocks])
    df_summary['participant'] = participants
    # Fill table with boredom score in each block
    for participant_index,participant in enumerate(participants):
        df_this = df_boredom_probes.loc[df_boredom_probes.participant==participant,:]
        for block in blocks:
            boredom_score = np.sum(df_this.loc[df_this.iBlock==block,'rating'])
            df_summary.loc[participant_index,f'block{block}'] = boredom_score
    # Return results
    return df_summary


# Calculate Cohen's D
def GetCohensD(x,y):
    # calculate the effect size (Cohen's D) given readings from
    # x (treatment group) and y (control group).

    # calculate d without weighting stddevs according to number of elements
    # (simpler if numel are the same in x and y)
    if len(x)==len(y):
        d = (np.mean(x) - np.mean(y)) / np.sqrt((np.std(x,ddof=1) ** 2 + np.std(y,ddof=1) ** 2) / 2)
        return d
    # weight stddevs according to number of elements
    else:
        # calculate intermediate values
        x_count = len(x)
        y_count = len(y)
        dof = x_count + y_count - 2 # degrees of freedom
        # calculate & return cohen's D
        d = (np.mean(x) - np.mean(y)) / np.sqrt(((x_count-1)*np.std(x, ddof=1) ** 2 + (y_count-1)*np.std(y, ddof=1) ** 2) / dof)
        return d

# Perform two one-sided t-tests to check if effect size magnitude of treatment is within specified bounds.
def TestMeanNeededForD(x,y,D=0.5):
    # check whether the mean of X (treatment group) given Y (control group)
    # indicates an effect size (Cohen's D) that is significantly -|D|<actualD<|D|.

    # calculate d without weighting stddevs according to number of elements
    # (simpler if numel are the same in x and y)
    if len(x)==len(y):
        # get limits
        meanX_lower = np.mean(y) - np.abs(D) * np.sqrt( (np.std(x,ddof=1)**2 + np.std(y,ddof=1) **2) / 2)
        meanX_upper = np.mean(y) + np.abs(D) * np.sqrt( (np.std(x,ddof=1)**2 + np.std(y,ddof=1) **2) / 2)
        dof = len(x)+len(y)-2

    # weight stddevs according to number of elements
    else:
        # get in-between quantities
        x_count = len(x)
        y_count = len(y)
        dof = x_count + y_count - 2 # degrees of freedom
        # get limits
        meanX_lower = np.mean(y) - np.abs(D) * np.sqrt( ((x_count-1)*np.std(x,ddof=1)**2 + (y_count-1)*np.std(x,ddof=1)**2) / dof) # mean of X corresponding to d_actual=-|D|
        meanX_upper = np.mean(y) + np.abs(D) * np.sqrt( ((x_count-1)*np.std(x,ddof=1)**2 + (y_count-1)*np.std(x,ddof=1)**2) / dof) # mean of X corresponding to d_actual=|D|

    # Run 2 one-sided t-tests
    t_more,p_more = stats.ttest_1samp(x,popmean=meanX_lower,alternative='greater') # H1: d_actual>-|D|. H0: d_actual<=-|D|.
    t_less,p_less = stats.ttest_1samp(x,popmean=meanX_upper,alternative='less') # H1: d_actual<|D|. H0: d_actual>=|D|.

    # return results
    return t_more,p_more,t_less,p_less,dof

# Print effect that a change of 1std would have on mood slope
def PrintEffectOf1StdChange(pymer_input,dfFit_h1,new_factor):
    mean_val = np.mean(pymer_input.loc[:,new_factor])
    std_val = np.std(pymer_input.loc[:,new_factor])
    before = (dfFit_h1.loc['Time','Estimate'] + dfFit_h1.loc[f'Time:{new_factor}','Estimate'] * mean_val) * 100
    change = (dfFit_h1.loc[f'Time:{new_factor}','Estimate'] * std_val) * 100
    after = before+change
    print(f'** An increase in {new_factor} of 1 std ({std_val:.03g}) from the mean ({mean_val:.03g}) \n'+
          f'** would change the estimated mood slope by {change:.03g} %mood/min, \n'+
          f'** from {before:.03g} to {after:.03g}, a change of {change/before*100:.03g}%.')

# Correlate new factor in fixed-effects pymer model with LME Time factor from reduced model
def PrintFactorSlopeCorrelations(dfFixef_h0,new_factor,factor_name,cohort_name='unknown'):
    print('')
    print(f'Correlating {factor_name} with LME slope in reduced model:')
    lme_slope = dfFixef_h0.Time.values
    MakeJointPlot(new_factor,lme_slope,factor_name,'lme_slope',cohort_name=cohort_name)
    # r,p = stats.pearsonr(new_factor,lme_slope)
    # print(f'Pearson r2={r**2:.3g},p={p:.3g}')
    # r_s,p_s = stats.spearmanr(new_factor,lme_slope)
    # print(f'Spearman r2={r_s**2:.3g},p={p_s:.3g}')
    # print('')

# Get
def GetBeforeAndAfterBoredom(df_boredom,pymer_input):
    # Set up
    participants = np.unique(df_boredom.participant)
    initial_boredom = np.zeros(len(participants))
    final_boredom = np.zeros(len(participants))
    delta_boredom = np.zeros(len(participants))
    # Loop through subjects
    for participant_index,participant in enumerate(participants):
        # crop to this participant
        df_this = df_boredom.loc[df_boredom.participant==participant,:]
        # get change in boredom scores
        initial_boredom[participant_index] = np.sum(df_this.loc[df_this.iBlock==-1,'rating']) # after first block
        final_boredom[participant_index] = np.sum(df_this.loc[df_this.iBlock==0,'rating']) # after first block
        delta_boredom[participant_index] = final_boredom[participant_index] - initial_boredom[participant_index]
        # add to pymer_input table
        pymer_input.loc[pymer_input.Subject==participant,'initialBoredom'] = initial_boredom[participant_index]
        pymer_input.loc[pymer_input.Subject==participant,'finalBoredom'] = final_boredom[participant_index]
        pymer_input.loc[pymer_input.Subject==participant,'deltaBoredom'] = delta_boredom[participant_index]


    df_summary = pd.DataFrame({'participant':participants,
                               'initial_boredom':initial_boredom,
                               'final_boredom':final_boredom,
                               'delta_boredom':delta_boredom})
    return pymer_input,df_summary


# Get last-mood minus first-mood for each participant in a list
def GetDeltaMood(pymer_input,participants):
    # get last-mood minus first-mood
    delta_mood = np.zeros(len(participants))
    for participant_index, participant in enumerate(participants):
        # pull out 1st-vs-last mood
        mood = pymer_input.loc[pymer_input.Subject==participant,'Mood'].values
        delta_mood[participant_index] = mood[-1]-mood[0]
    return delta_mood

# Make a Seaborn joint plot with a regression line, printing stats for each factor and the regression
def MakeJointPlot(stat_a,stat_b,stat_a_name,stat_b_name,cohort_name='unknown'):
    print(f'= {stat_a_name}:')
    D = np.mean(stat_a)/np.std(stat_a)
    print(f' mean={np.mean(stat_a):.3g}, std={np.std(stat_a):.3g}, D={D:.3g}')
    t,p = stats.ttest_1samp(stat_a,0)
    print(f' 2-sided t-test against 0: t={t:.3g},p={p:.3g}')

    print(f'= {stat_b_name}:')
    D = np.mean(stat_b)/np.std(stat_b)
    print(f' mean={np.mean(stat_b):.3g}, std={np.std(stat_b):.3g}, D={D:.3g}')
    t,p = stats.ttest_1samp(stat_b,0)
    print(f' {stat_b_name} ~=0: t={t:.3g},p={p:.3g}')


    print('= Correlation:')
    r,p = stats.pearsonr(stat_a,stat_b)
    print(f' Pearson r2={r**2:.3g},p={p:.3g}')
    r_s,p_s = stats.spearmanr(stat_a,stat_b)
    print(f' Spearman r2={r_s**2:.3g},p={p_s:.3g}')

    # Do joint plot
    df_stat = pd.DataFrame()
    df_stat[stat_a_name] = stat_a
    df_stat[stat_b_name] = stat_b
    sns.jointplot(x=stat_a_name,y=stat_b_name,data=df_stat,kind="reg")
    # annotate plot
    plt.suptitle(f'Cohort {cohort_name}: {stat_a_name} vs. {stat_b_name}\n'+
                 f'r_s^2={r_s**2:.3g},p_s={p_s:.3g}')
    plt.tight_layout()
    plt.subplots_adjust(top=0.90) # Reduce plot to make room
    # save plot
    fig_file = f'{figures_dir}/{cohort_name}_{stat_a_name}-vs-{stat_b_name}_jointplot.png'
    print(f'=Saving {stat_a_name} vs. {stat_b_name} jointplot as {fig_file}....')
    plt.savefig(fig_file)
    fig_file = f'{figures_dir}/{cohort_name}_{stat_a_name}-vs-{stat_b_name}_jointplot.pdf'
    print(f'=Saving {stat_a_name} vs. {stat_b_name} jointplot as {fig_file}....')
    plt.savefig(fig_file)


# Crop to exclude all mood ratings after miniumum rating (for floor effects)
def CropToMinRating(pymer_input):
    # get list of subjects
    participants = np.unique(pymer_input.Subject)
    for participant_index,participant in enumerate(participants):
        # crop to this subject's data
        df_this = pymer_input.loc[pymer_input.Subject==participant,:]
        min_index = np.argmin(df_this['Mood']) # find index of minimum rating
        pymer_input = pymer_input.drop(df_this.index[min_index+1:])
    return pymer_input

# Declare file locations
results_dir = '../Data/OutFiles'
figures_dir = '../Figures' # where figures should be saved

# %% Demographics of new cohorts

in_file = f'{results_dir}/Mmi-Batches.csv'
df_batches = pd.read_csv(in_file)

is_control_batch = df_batches.endDate>'2021-01-01'
subj_attempted_count = np.sum(df_batches.loc[is_control_batch,'nSubjAttempted'])
subj_completed_count = np.sum(df_batches.loc[is_control_batch,'nSubjCompleted'])

print('=======================================')
print('=== DEMOGRAPHICS ===')

print(f'{subj_attempted_count} participants completed these tasks online. ' +
      f'{subj_attempted_count-subj_completed_count} participants were ' +
      'excluded because their task or survey data was incomplete or did not ' +
      'save, because they completed the task more than once despite ' +
      'instructions to the contrary, or because they failed to answer one ' +
      'or more "catch" questions correctly on the survey.')

# get batch names
batches = df_batches.loc[is_control_batch,'batchName']
# batches = ['Activities','AllMw','AllBoredom']

# Get age/gender info
participant_count = 0
female_count = 0
age_sum = 0
max_age = 0
min_age = 99
for batch in batches:
    in_file = f'{results_dir}/Mmi-{batch}_Survey.csv'
    df_survey = pd.read_csv(in_file)
    participant_count += df_survey.shape[0]
    female_count += np.sum(df_survey.gender=='Female')
    age_sum += np.sum(df_survey.age)
    max_age = np.max([max_age,np.max(df_survey.age)])
    min_age = np.min([min_age,np.min(df_survey.age)])

# print results
print(f'Of the {participant_count} remaining participants, {female_count} ' +
      f'were female ({female_count/participant_count*100:.3g}%). Participants ' +
      f'had a mean age of {age_sum/participant_count:.3g} years (range: {min_age:.0f}-{max_age:.0f}).')

# %% Hyp 1.1: Boredom Repeat Administraion
print('=======================================')
print('')
print('=======================================')
print("""
Boredom Hypotheses:
1.1) In the validation of short-interval state boredom scale repeat
    administration,we hypothesize that the effect of including an initial
    administration will have an absolute effect size (cohen’s d) less than 0.5.
    We will test this with two, one-sided t-tests (TOST).
""")
print('=== BOREDOM REPEAT ADMINISTRATION ===')


# Declare batches
batch_ba = 'BoredomBeforeAndAfter' # before-and-after group, got thought probes both before and after rest block
batch_ao = 'BoredomAfterOnly' # after-only group, got thought probes only after rest block

# Before and after group: Load probes file
in_file = f'{results_dir}/Mmi-{batch_ba}_Probes.csv'
df_boredom_probes_ba = pd.read_csv(in_file)
# Get boredom scores in each block
df_summary_ba = GetBoredomScores(df_boredom_probes_ba)

# After-only group: Load probes file
in_file = f'{results_dir}/Mmi-{batch_ao}_Probes.csv'
df_boredom_probes_ao = pd.read_csv(in_file)
# Get boredom scores in each block
df_summary_ao = GetBoredomScores(df_boredom_probes_ao)

# Perform t-tests to check for differences between groups
block_names = df_summary_ao.columns[1:]
for block_to_check in block_names:
    print(f'= {block_to_check} =')

    # calculate cohen's d
    cohens_d = GetCohensD(df_summary_ba[block_to_check],df_summary_ao[block_to_check])
    # Check mean of before-and-after group against cohen's D from preregistration
    D_cutoff = 0.5
    t_more,p_more,t_less,p_less,dof = TestMeanNeededForD(df_summary_ba[block_to_check],df_summary_ao[block_to_check],D=D_cutoff)
    # Print results
    print(f'{batch_ba} vs. {batch_ao}: Cohens D={cohens_d:.03g}')
    print(f'Is {batch_ba} < {batch_ao} with Cohens D>{-np.abs(D_cutoff):.03g}: T_{dof}={t_more:.03g}, p={p_more:.03g}')
    print(f'Is {batch_ba} > {batch_ao} with Cohens D<{np.abs(D_cutoff):.03g}: T_{dof}={t_less:.03g}, p={p_less:.03g}')


    if block_to_check=='block0':
        # Print conclusions
        if (p_less<0.05) and (p_more>0.05):
            print(f'** Presenting boredom questions before start of task leads to DECREASED responses after {block_to_check}.')
            use_both_boredom = False
            print(f'** because we cannot exclude H0:|D|>={np.abs(D_cutoff):.03g}, we will use only the {batch_ao} cohort in subsequent analyses.')
        elif (p_more<0.05) and (p_less>0.05):
            print(f'** Presenting boredom questions before start of task leads to INCREASED responses after {block_to_check}.')
            use_both_boredom = False
            print(f'** because we cannot exclude H0:|D|>={np.abs(D_cutoff):.03g}, we will use only the {batch_ao} cohort in subsequent analyses.')
        else:
            print(f'** Presenting boredom questions before start of task DOES NOT change responses after {block_to_check}.')
            use_both_boredom = True
            print(f'** because we can exclude H0:|D|>{np.abs(D_cutoff):.03g}, we will use both boredom cohorts in subsequent analyses.')



# %% Hyp 1.2: Effect of finalBoredom on mood
print('=======================================')
print('')
print('=======================================')
print("""
1.2) We hypothesize that final state boredom will explain variance in
    subject-level POTD slope. This is a one-sided hypothesis.
    We will test this with an ANOVA comparing the following two mixed effects
    models (difference highlighted in bold):
    H0: Mood ~ 1 + Time * (isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)
    H1: Mood ~ 1 + Time * (finalBoredom + isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)
""")


# If repeat administration changes results, we'll use the after-only group.
# Otherwise, use both.
if use_both_boredom:
    batch = 'AllBoredom'
else:
    batch = 'BoredomAfterOnly'

print(f'=== Batch {batch}: Comparing LME models with and without finalBoredom ===')
# Load pymer input file
in_file = f'{results_dir}/Mmi-{batch}_pymerInput-full.csv'
print(f'Opening {in_file}...')
pymer_input = pd.read_csv(in_file, index_col=0)

anova_res = None
for do_premin_only in [False,True]:

    # Control for floor effects
    if do_premin_only:
        if anova_res.loc[1,'Pr(>Chisq)']<0.05:
            print('=======================================\n\n'+
                  '=======================================')
            print('=== 4.1.2: Control for floor effects by using excluding ratings after a subject''s minimum')
            # Crop to exclude all mood ratings after miniumum rating (for floor effects)
            pymer_input = CropToMinRating(pymer_input)
            cohort_name = f'{batch}-premin'
        else:
            print('=== Skipping floor effects control because original was not significant.')
            break
    else:
        cohort_name = batch

    # Add finalBoredom scores
    in_file = f'{results_dir}/Mmi-{batch}_Probes.csv'
    print(f'Opening {in_file}...')
    df_boredom = pd.read_csv(in_file)
    participants = np.unique(df_boredom.participant)
    final_boredom = np.zeros(len(participants))
    for participant_index,participant in enumerate(participants):
        # crop to this participant
        df_this = df_boredom.loc[df_boredom.participant==participant,:]
        # get final boredom score
        final_boredom[participant_index] = np.sum(df_this.loc[df_this.iBlock==0,'rating']) # after first block
        # add to pymer_input table
        pymer_input.loc[pymer_input.Subject==participant,'finalBoredom'] = final_boredom[participant_index]

    # Plot stat vs. change in mood
    delta_mood = GetDeltaMood(pymer_input,participants)
    MakeJointPlot(final_boredom,delta_mood,'final_boredom','delta_mood',cohort_name=cohort_name)

    # Fit models and run ANOVA to compare
    lm_string_h0 = 'Mood ~ 1 + Time * (isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)'
    lm_string_h1 = 'Mood ~ 1 + Time * (finalBoredom + isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)'
    anova_res, dfFit_h0, dfFit_h1, dfFixef_h0 = compare_lmers(pymer_input,lm_string_h0,lm_string_h1)
    print('ANOVA:')
    print(anova_res)

    # correlate new factor with subject LME slopes in reduced model
    PrintFactorSlopeCorrelations(dfFixef_h0,final_boredom,'finalBoredom',cohort_name)

    # Print results and pymer fit
    if anova_res.loc[1,'Pr(>Chisq)']<0.05:
        print('** Final state boredom DOES explain added variance in subject-level POTD slope.')
    else:
        print('** Final state boredom does NOT explain added variance in subject-level POTD slope.')
    print(dfFit_h1.loc[['Time','finalBoredom','Time:finalBoredom']])

    # Print effect that a change of 1std would have on mood slope
    PrintEffectOf1StdChange(pymer_input,dfFit_h1,'finalBoredom')


# %% Hyp 1.3: Effect of deltaBoredom on mood
print('=======================================')
print('')
print('=======================================')
print("""
1.3) We hypothesize that the change in boredom will explain variance in
    subject-level POTD slope. This is a one-sided hypothesis.
    We will test this with an ANOVA comparing the following two mixed effects
    models (difference highlighted in bold):
    H0: Mood ~ 1 + Time * (isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)
    H1: Mood ~ 1 + Time * (deltaBoredom + isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)
    If we fail to reject the null for hypothesis 1.1 (absolute cohen’s d is less
    than 0.5) we will have to interpret the results of this hypothesis with the
    caveat that it is possible that repeated administration of the state
    boredom measure may have altered the results of the subsequent administration.
""")

# Analyzing change in boredom requires before-and-after group
batch = 'BoredomBeforeAndAfter'

print(f'=== Batch {batch}: Comparing LME models with and without deltaBoredom ===')
# Load pymer input file
in_file = f'{results_dir}/Mmi-{batch}_pymerInput-full.csv'
print(f'Opening {in_file}...')
pymer_input = pd.read_csv(in_file, index_col=0)

anova_res = None
for do_premin_only in [False,True]:

    # Control for floor effects
    if do_premin_only:
        if anova_res.loc[1,'Pr(>Chisq)']<0.05:
            print('=======================================\n\n'+
                  '=======================================')
            print('=== 4.1.3: Control for floor effects by using excluding ratings after a subject''s minimum')
            # Crop to exclude all mood ratings after miniumum rating (for floor effects)
            pymer_input = CropToMinRating(pymer_input)
            cohort_name = f'{batch}-premin'
        else:
            print('=== Skipping floor effects control because original was not significant.')
            break
    else:
        cohort_name = batch

    # Add deltaBoredom scores
    in_file = f'{results_dir}/Mmi-{batch}_Probes.csv'
    print(f'Opening {in_file}...')
    df_boredom = pd.read_csv(in_file)
    pymer_input,df_summary = GetBeforeAndAfterBoredom(df_boredom,pymer_input)

    # Plot stat vs. change in mood
    delta_mood = GetDeltaMood(pymer_input,df_summary['participant'])
    MakeJointPlot(df_summary['delta_boredom'],delta_mood,'delta_boredom','delta_mood',cohort_name=cohort_name)
    # for completeness, also do initial & final for this group
    MakeJointPlot(df_summary['initial_boredom'],delta_mood,'initial_boredom','delta_mood',cohort_name=cohort_name)
    MakeJointPlot(df_summary['final_boredom'],delta_mood,'final_boredom','delta_mood',cohort_name=cohort_name)

    # Fit models and run ANOVA to compare
    lm_string_h0 = 'Mood ~ 1 + Time * (isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)'
    lm_string_h1 = 'Mood ~ 1 + Time * (deltaBoredom + isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)'
    anova_res, dfFit_h0, dfFit_h1, dfFixef_h0 = compare_lmers(pymer_input,lm_string_h0,lm_string_h1)
    print('ANOVA:')
    print(anova_res)

    # correlate new factor with subject LME slopes in reduced model
    PrintFactorSlopeCorrelations(dfFixef_h0,df_summary['delta_boredom'],'deltaBoredom',cohort_name)

    # Print results and pymer fit
    if anova_res.loc[1,'Pr(>Chisq)']<0.05:
        print('** Change in state boredom DOES explain added variance in subject-level POTD slope.')
    else:
        print('** Change in state boredom does NOT explain added variance in subject-level POTD slope.')
    print(dfFit_h1.loc[['Time','deltaBoredom','Time:deltaBoredom']])

    # Print effect that a change of 1std would have on mood slope
    PrintEffectOf1StdChange(pymer_input,dfFit_h1,'deltaBoredom')

# %% Hyp 1.4: Effect of traitBoredom on mood
print('=======================================')
print('')
print('=======================================')
print("""
1.4) We hypothesize that trait boredom will explain variance in subject-level
    POTD slope.This is a one-sided hypothesis.
    We will test this with an ANOVA comparing the following two mixed effects
    models (difference highlighted in bold):
    H0: Mood ~ 1 + Time * (isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)
    H1: Mood ~ 1 + Time * (traitBoredom + isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)
""")


# If repeat administration changes results, we'll use the after-only group.
# Otherwise, use both.
if use_both_boredom:
    batch = 'AllBoredom'
else:
    batch = 'BoredomAfterOnly'

print(f'=== Batch {batch}: Comparing LME models with and without traitBoredom ===')
# Load pymer input file
in_file = f'{results_dir}/Mmi-{batch}_pymerInput-full.csv'
print(f'Opening {in_file}...')
pymer_input = pd.read_csv(in_file, index_col=0)

anova_res = None
for do_premin_only in [False,True]:

    # Control for floor effects
    if do_premin_only:
        if anova_res.loc[1,'Pr(>Chisq)']<0.05:
            print('=======================================\n\n'+
                  '=======================================')
            print('=== 4.1.4: Control for floor effects by using excluding ratings after a subject''s minimum')
            # Crop to exclude all mood ratings after miniumum rating (for floor effects)
            pymer_input = CropToMinRating(pymer_input)
            cohort_name = f'{batch}-premin'
        else:
            print('=== Skipping floor effects control because original was not significant.')
            break
    else:
        cohort_name = batch

    # Add traitBoredom scores
    in_file = f'{results_dir}/Mmi-{batch}_Survey.csv'
    print(f'Opening {in_file}...')
    df_boredom = pd.read_csv(in_file)
    participants = np.unique(df_boredom.participant)
    trait_boredom = np.zeros(len(participants))
    delta_mood = GetDeltaMood(pymer_input,participants)
    for participant_index,participant in enumerate(participants):
        trait_boredom[participant_index] = df_boredom.loc[df_boredom.participant==participant,'BORED'].values[0]
        pymer_input.loc[pymer_input.Subject==participant,'traitBoredom'] = trait_boredom[participant_index]

    # Fit models and run ANOVA to compare
    lm_string_h0 = 'Mood ~ 1 + Time * (isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)'
    lm_string_h1 = 'Mood ~ 1 + Time * (traitBoredom + isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)'
    anova_res, dfFit_h0, dfFit_h1, dfFixef_h0 = compare_lmers(pymer_input,lm_string_h0,lm_string_h1)
    print('ANOVA:')
    print(anova_res)

    # correlate new factor with subject LME slopes in reduced model
    PrintFactorSlopeCorrelations(dfFixef_h0,trait_boredom,'traitBoredom',cohort_name)

    # Print results and pymer fit
    if anova_res.loc[1,'Pr(>Chisq)']<0.05:
        print('** Trait boredom DOES explain added variance in subject-level POTD slope.')
    else:
        print('** Trait boredom does NOT explain added variance in subject-level POTD slope.')
    print(dfFit_h1.loc[['Time','traitBoredom','Time:traitBoredom']])

    # Print effect that a change of 1std would have on mood slope
    PrintEffectOf1StdChange(pymer_input,dfFit_h1,'traitBoredom')

# %% Get MW principal components
print('=======================================')
print('')
print('=======================================')

print("""
Mind Wandering Hypotheses:
2.1) In the validation of short-interval MDES repeat administration, we
    hypothesize that the effect of including an initial administration will
    have an absolute effect size (cohen’s d) less than 0.5.
    We will test this with two, one-sided t-tests (TOST).
""")

print('=== MW PCA ===' )

# Get all probes and run PCA
# batch = 'MwBeforeAndAfter' # before-and-after group
batch = 'MwAfterOnly' # after-only group
# Load probes file
in_file = f'{results_dir}/Mmi-{batch}_Probes.csv'
df_mw = pd.read_csv(in_file)
# Extract ratings and center scale at 0
X = df_mw['rating'].values.reshape([-1,13])-0.5
# Fit PCA to these ratings
pca = PCA(n_components=13,whiten=True)
pca.fit(X)
# print(pca.explained_variance_ratio_)
# print(pca.singular_values_)

# make plot of variance explained
fig = plt.figure(23,clear=True)
plt.plot(np.cumsum(pca.explained_variance_ratio_)*100)
plt.xlabel('component')
plt.ylabel('% variance explained')
plt.title('MW probe PCA')
# Save figure
fig_file = f'{figures_dir}/{batch}_MwPca_VarExplained.png'
print(f'Saving figure as {fig_file}...')
fig.savefig(fig_file)
fig_file = f'{figures_dir}/{batch}_MwPca_VarExplained.pdf'
print(f'Saving figure as {fig_file}...')
fig.savefig(fig_file)

# === Plot PC loadings
# Set up figure
pc_count = pca.n_components
question_labels = np.array(['task','future','past','myself','people','emotion','images','words','vivid','detailed','habit','evolving','deliberate'])
ticks = np.arange(len(question_labels))
fig,axes = plt.subplots(4,4,num=24,figsize=[12,8],clear=True,sharex=False,sharey=True)
axes = axes.flatten()
# Plot bars of loadings
for plot_index,ax in enumerate(axes):
    if plot_index<pc_count:
        # make bar plot
        ax.bar(ticks,pca.components_[plot_index,:])
        # set title
        variance_explained = pca.explained_variance_ratio_[plot_index]*100
        ax.set_title(f'Comp {plot_index}: varex={variance_explained:.1f}')
        # annotate plot
        ax.grid(True)
        ax.set_xlabel('question')
        ax.set_ylabel('loading')
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels=question_labels,ha='right',rotation=45)
    else:
        # remove extra plots
        ax.set_visible(False)
plt.tight_layout()

# Save figure
fig_file = f'{figures_dir}/{batch}_MwPcaLoadings.png'
print(f'Saving figure as {fig_file}...')
fig.savefig(fig_file)
fig_file = f'{figures_dir}/{batch}_MwPcaLoadings.pdf'
print(f'Saving figure as {fig_file}...')
fig.savefig(fig_file)

# Note the most emotion-related PC
# defined as the one with the largest magnitude loading on the emotion question
emotion_pc_index = np.argmax(np.abs(pca.components_[:,question_labels=='emotion'])) # 4
print(f'PC #{emotion_pc_index} appears to be emotion component.')


# %%  Hyp 2.1: MW emotion repeat administration
# get summary scores
print('===')
print('')
print('=== MW REPEAT ADMINISTRATION ===')

# Function to get summary boredom scores for each participant
def GetMwScores(df_mw_probes,pca):
    # Extract unique list of participant & blocks
    participants = np.unique(df_mw_probes.participant)
    blocks = np.unique(df_mw_probes.iBlock).astype(int)
    # Set up summary table
    df_summary = pd.DataFrame(columns=['participant']+[f'block{x}' for x in blocks])
    df_summary['participant'] = participants
    # Fill table with boredom score in each block
    for participant_index,participant in enumerate(participants):
        df_this = df_mw_probes.loc[df_mw_probes.participant==participant,:]
        for block in blocks:
            mw_ratings = np.atleast_2d(df_this.loc[df_this.iBlock==block,'rating'])-0.5 # make 1x13, move center of scale to 0
            mw_score = pca.transform(mw_ratings)[0,emotion_pc_index] # use pca transformation to get MW score
            df_summary.loc[participant_index,f'block{block}'] = mw_score

    return df_summary


batch_ba = 'MwBeforeAndAfter'
batch_ao = 'MwAfterOnly'

in_file = f'{results_dir}/Mmi-{batch_ba}_Probes.csv'
df_mw_ba = pd.read_csv(in_file)
df_summary_ba = GetMwScores(df_mw_ba,pca)


in_file = f'{results_dir}/Mmi-{batch_ao}_Probes.csv'
df_mw_ao = pd.read_csv(in_file)
df_summary_ao = GetMwScores(df_mw_ao,pca)

block_names = df_summary_ao.columns[1:]
for block_to_check in block_names:
    print(f'= {block_to_check} =')

    # calculate cohen's d
    cohens_d = GetCohensD(df_summary_ba[block_to_check],df_summary_ao[block_to_check])
    # Check mean of before-and-after group against cohen's D from preregistration
    D_cutoff = 0.5
    t_more,p_more,t_less,p_less,dof = TestMeanNeededForD(df_summary_ba[block_to_check],df_summary_ao[block_to_check],D=D_cutoff)
    # Print results
    print(f'{batch_ba} vs. {batch_ao}: Cohens D={cohens_d:.03g}')
    print(f'Is {batch_ba} < {batch_ao} with Cohens D>{-np.abs(D_cutoff):.03g}: T_{dof}={t_more:.03g}, p={p_more:.03g}')
    print(f'Is {batch_ba} > {batch_ao} with Cohens D<{np.abs(D_cutoff):.03g}: T_{dof}={t_less:.03g}, p={p_less:.03g}')


    if block_to_check=='block0':
        # Print conclusions
        if (p_less<0.05) and (p_more>0.05):
            print(f'** Presenting MW questions before start of task leads to DECREASED responses after {block_to_check}.')
            use_both_mw = False
            print(f'** because we cannot exclude H0:|D|>={np.abs(D_cutoff):.03g}, we will use only the {batch_ao} cohort in subsequent analyses.')
        elif (p_more<0.05) and (p_less>0.05):
            print(f'** Presenting MW questions before start of task leads to INCREASED responses after {block_to_check}.')
            use_both_mw = False
            print(f'** because we cannot exclude H0:|D|>={np.abs(D_cutoff):.03g}, we will use only the {batch_ao} cohort in subsequent analyses.')
        else:
            print(f'** Presenting MW questions before start of task DOES NOT change responses after {block_to_check}.')
            use_both_mw = True
            print(f'** because we can exclude H0:|D|>{np.abs(D_cutoff):.03g}, we will use both MW cohorts in subsequent analyses.')




# %% Hyp 2.2: Effect of finalEmoDim on mood
print('=======================================')
print('')
print('=======================================')
print("""
2.2) We hypothesize that the final emotion dimension score will explain
    variance in subject-level POTD slope. This is a one-sided hypothesis.
    We will test this with an ANOVA comparing the following two mixed effects
    models (difference highlighted in bold):
    H0: Mood ~ 1 + Time * (isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)
    H1: Mood ~ 1 + Time * (finalEmoDim + isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)
""")

# If repeat administration changes results, we'll use the after-only group.
# Otherwise, use both.
if use_both_mw:
    batch = 'AllMw'
else:
    batch = 'MwAfterOnly'

print(f'=== Batch {batch}: Comparing LME models with and without finalEmoDim ===')
in_file = f'{results_dir}/Mmi-{batch}_pymerInput-full.csv'
print(f'Opening {in_file}...')
pymer_input = pd.read_csv(in_file, index_col=0)

anova_res = None
for do_premin_only in [False,True]:

    # Control for floor effects
    if do_premin_only:
        if anova_res.loc[1,'Pr(>Chisq)']<0.05:
            print('=======================================\n\n'+
                  '=======================================')
            print('=== 4.2.2: Control for floor effects by using excluding ratings after a subject''s minimum')
            # Crop to exclude all mood ratings after miniumum rating (for floor effects)
            pymer_input = CropToMinRating(pymer_input)
            cohort_name = f'{batch}-premin'
        else:
            print('=== Skipping floor effects control because original was not significant.')
            break
    else:
        cohort_name = batch

    # Add finalEmoDim scores
    in_file = f'{results_dir}/Mmi-{batch}_Probes.csv'
    print(f'Opening {in_file}...')
    df_mw = pd.read_csv(in_file)
    participants = np.unique(df_mw.participant)
    final_mw = np.zeros(len(participants))
    delta_mood = GetDeltaMood(pymer_input,participants)
    for participant_index,participant in enumerate(participants):
        # crop to this participant
        df_this = df_mw.loc[df_mw.participant==participant,:]
        # get final MW score
        X_this = np.atleast_2d(df_this.loc[df_this.iBlock==0,'rating'])-0.5 # iBlock==0: after first block. -0.5: Move center of scale to 0
        final_mw[participant_index] = pca.transform(X_this)[0,emotion_pc_index]
        # Add to pymer_input table
        pymer_input.loc[pymer_input.Subject==participant,'finalEmoDim'] = final_mw[participant_index]


    # Plot stat vs. change in mood
    delta_mood = GetDeltaMood(pymer_input,participants)
    MakeJointPlot(final_mw,delta_mood,'final_mw','delta_mood',cohort_name=cohort_name)

    # Fit models and run ANOVA to compare
    lm_string_h0 = 'Mood ~ 1 + Time * (isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)'
    lm_string_h1 = 'Mood ~ 1 + Time * (finalEmoDim + isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)'
    anova_res, dfFit_h0, dfFit_h1, dfFixef_h0 = compare_lmers(pymer_input,lm_string_h0,lm_string_h1)
    print('ANOVA:')
    print(anova_res)

    # correlate new factor with subject LME slopes in reduced model
    PrintFactorSlopeCorrelations(dfFixef_h0,final_mw,'finalEmoDim',cohort_name)

    # Print results and fit
    if anova_res.loc[1,'Pr(>Chisq)']<0.05:
        print('** Final MW emotion DOES explain added variance in subject-level POTD slope.')
    else:
        print('** Final MW emotion does NOT explain added variance in subject-level POTD slope.')
    print(dfFit_h1.loc[['Time','finalEmoDim','Time:finalEmoDim']])

    # Print effect that a change of 1std would have on mood slope
    PrintEffectOf1StdChange(pymer_input,dfFit_h1,'finalEmoDim')


# %% Hyp 2.3: Effect of deltaEmoDim on mood
print('=======================================')
print('')
print('=======================================')
print("""
2.3) We hypothesize that the change in emotion dimension score will explain
    variance in subject-level POTD slope. This is a one-sided hypothesis.
    We will test this with an ANOVA comparing the following two mixed effects
    models (difference highlighted in bold):
    H0: Mood ~ 1 + Time * (isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)
    H1: Mood ~ 1 + Time * (deltaEmoDim + isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)
    If we fail to reject the null for hypothesis 2.1 (absolute cohen’s d is
    less than 0.5) we will have to interpret the results of this hypothesis
    with the caveat that it is possible that repeated administration of the
    MDES measure may have altered the results of the subsequent administration.
""")

# Analyzing change requires before-and-after batch
batch = 'MwBeforeAndAfter'

print(f'=== Batch {batch}: Comparing LME models with and without deltaEmoDim ===')
# Load pymyer input file
in_file = f'{results_dir}/Mmi-{batch}_pymerInput-full.csv'
print(f'Opening {in_file}...')
pymer_input = pd.read_csv(in_file, index_col=0)

anova_res = None
for do_premin_only in [False,True]:

    # Control for floor effects
    if do_premin_only:
        if anova_res.loc[1,'Pr(>Chisq)']<0.05:
            print('=======================================\n\n'+
                  '=======================================')
            print('=== 4.2.3: Control for floor effects by using excluding ratings after a subject''s minimum')
            # Crop to exclude all mood ratings after miniumum rating (for floor effects)
            pymer_input = CropToMinRating(pymer_input)
            cohort_name = f'{batch}-premin'
        else:
            print('=== Skipping floor effects control because original was not significant.')
            break
    else:
        cohort_name = batch

    # Add deltaEmoDim scores
    in_file = f'{results_dir}/Mmi-{batch}_Probes.csv'
    print(f'Opening {in_file}...')
    df_mw = pd.read_csv(in_file)
    participants = np.unique(df_mw.participant)
    initial_mw = np.zeros(len(participants))
    final_mw = np.zeros(len(participants))
    delta_mw = np.zeros(len(participants))
    delta_mood = GetDeltaMood(pymer_input,participants)
    for participant_index,participant in enumerate(participants):
        # crop to this participant
        df_this = df_mw.loc[df_mw.participant==participant,:]
        # get initial MW score
        X_initial = np.atleast_2d(df_this.loc[df_this.iBlock==-1,'rating'])-0.5 # Before 1st block. Move center of scale to 0
        initial_mw[participant_index] = pca.transform(X_initial)[0,emotion_pc_index]
        # get final MW score
        X_final = np.atleast_2d(df_this.loc[df_this.iBlock==0,'rating'])-0.5 # After 1st block. Move center of scale to 0
        final_mw[participant_index] = pca.transform(X_final)[0,emotion_pc_index]
        # Add to pymer_input table
        delta_mw[participant_index] = final_mw[participant_index] - initial_mw[participant_index]
        pymer_input.loc[pymer_input.Subject==participant,'deltaEmoDim'] = delta_mw[participant_index]


    # Plot stat vs. change in mood
    delta_mood = GetDeltaMood(pymer_input,participants)
    MakeJointPlot(delta_mw,delta_mood,'delta_mw','delta_mood',cohort_name=cohort_name)
    # for completeness, also do initial & final for this group
    MakeJointPlot(initial_mw,delta_mood,'initial_mw','delta_mood',cohort_name=cohort_name)
    MakeJointPlot(final_mw,delta_mood,'final_mw','delta_mood',cohort_name=cohort_name)

    # Fit models and run ANOVA to compare
    lm_string_h0 = 'Mood ~ 1 + Time * (isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)'
    lm_string_h1 = 'Mood ~ 1 + Time * (deltaEmoDim + isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)'
    anova_res, dfFit_h0, dfFit_h1, dfFixef_h0 = compare_lmers(pymer_input,lm_string_h0,lm_string_h1)
    print('ANOVA:')
    print(anova_res)

    # correlate new factor with subject LME slopes in reduced model
    PrintFactorSlopeCorrelations(dfFixef_h0,delta_mw,'deltaEmoDim',cohort_name)

    # Print results and fit
    if anova_res.loc[1,'Pr(>Chisq)']<0.05:
        print('** Change in MW emotion DOES explain added variance in subject-level POTD slope.')
    else:
        print('** Change in MW emotion does NOT explain added variance in subject-level POTD slope.')
    print(dfFit_h1.loc[['Time','deltaEmoDim','Time:deltaEmoDim']])

    # Print effect that a change of 1std would have on mood slope
    PrintEffectOf1StdChange(pymer_input,dfFit_h1,'deltaEmoDim')

# %% Hyp 2.4: Effect of traitMW on mood
print('=======================================')
print('')
print('=======================================')
print("""
2.4) We hypothesize that trait mind wandering will explain variance in
    subject-level POTD slope. This is a one-sided hypothesis.
    We will test this with an ANOVA comparing the following two mixed effects
    models (difference highlighted in bold):
    H0: Mood ~ 1 + Time * (isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)
    H1: Mood ~ 1 + Time * (traitMW + isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)
""")

# If repeat administration changes results, we'll use the after-only group.
# Otherwise, use both.
if use_both_mw:
    batch = 'AllMw'
else:
    batch = 'MwAfterOnly'

print(f'=== Batch {batch}: Comparing LME models with and without traitMW ===')
# load pymer input tables
in_file = f'{results_dir}/Mmi-{batch}_pymerInput-full.csv'
print(f'Opening {in_file}...')
pymer_input = pd.read_csv(in_file, index_col=0)

anova_res = None
for do_premin_only in [False,True]:

    # Control for floor effects
    if do_premin_only:
        if anova_res.loc[1,'Pr(>Chisq)']<0.05:
            print('=======================================\n\n'+
                  '=======================================')
            print('=== 4.2.4: Control for floor effects by using excluding ratings after a subject''s minimum')
            # Crop to exclude all mood ratings after miniumum rating (for floor effects)
            pymer_input = CropToMinRating(pymer_input)
            cohort_name = f'{batch}-premin'
        else:
            print('=== Skipping floor effects control because original was not significant.')
            break
    else:
        cohort_name = batch

    # Add traitMW scores
    in_file = f'{results_dir}/Mmi-{batch}_Survey.csv'
    print(f'Opening {in_file}...')
    df_mw = pd.read_csv(in_file)
    participants = np.unique(df_mw.participant)
    trait_mw = np.zeros(len(participants))
    delta_mood = GetDeltaMood(pymer_input,participants)
    for participant_index,participant in enumerate(participants):
        # get trait MW score from table
        trait_mw[participant_index] = df_mw.loc[df_mw.participant==participant,'MW'].values[0]
        # add to pymer_input table
        pymer_input.loc[pymer_input.Subject==participant,'traitMW'] = trait_mw[participant_index]

    # Plot stat vs. change in mood
    delta_mood = GetDeltaMood(pymer_input,participants)
    MakeJointPlot(trait_mw,delta_mood,'trait_mw','delta_mood',cohort_name=cohort_name)

    # Fit models and run ANOVA to compare
    lm_string_h0 = 'Mood ~ 1 + Time * (isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)'
    lm_string_h1 = 'Mood ~ 1 + Time * (traitMW + isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)'
    anova_res, dfFit_h0, dfFit_h1, dfFixef_h0 = compare_lmers(pymer_input,lm_string_h0,lm_string_h1)
    print('ANOVA:')
    print(anova_res)

    # correlate new factor with subject LME slopes in reduced model
    PrintFactorSlopeCorrelations(dfFixef_h0,trait_mw,'traitMW',cohort_name)

    # Print results and fit
    if anova_res.loc[1,'Pr(>Chisq)']<0.05:
        print('** Trait MW DOES explain added variance in subject-level POTD slope.')
    else:
        print('** Trait MW does NOT explain added variance in subject-level POTD slope.')
    print(dfFit_h1.loc[['Time','traitMW','Time:traitMW']])

    # Print effect that a change of 1std would have on mood slope
    PrintEffectOf1StdChange(pymer_input,dfFit_h1,'traitMW')

# %% Hyp 3.1: Effect of real-world free activities on mood
print('=======================================')
print('')
print('=======================================')
print("""
Real World Hypothesis:
3.1) We hypothesize that final mood ratings will be lower on average than the
    initial mood ratings in our real-world task.
    We will test this with a paired sample t-test between mood ratings prior
    to the break and mood ratings following the break.
""")

print('=== POTD WITH AGENCY ===')
batch = 'Activities'
# Load mood ratings
in_file = f'{results_dir}/Mmi-{batch}_Ratings.csv'
df_act = pd.read_csv(in_file)
# Create summary table of happiness scores
participants = np.unique(df_act.participant)
df_summary = pd.DataFrame(columns=['participant','happinessBefore','happinessAfter'])
df_summary['participant'] = participants
# Fill in with before-and-after happiness ratings
block_to_check = 0
for participant_index,participant in enumerate(participants):
    # extract happiness rating before and after break period
    df_this = df_act.loc[(df_act.participant==participant) & (df_act.iBlock==block_to_check),:]
    df_summary.loc[participant_index,'happinessBefore'] = df_this.rating.values[0]
    df_summary.loc[participant_index,'happinessAfter'] = df_this.rating.values[-1]

# Test stats with 2 one-sided t-tests
t_less,p_less = stats.ttest_rel(df_summary['happinessBefore'],df_summary['happinessAfter'],alternative='less')
t_more,p_more = stats.ttest_rel(df_summary['happinessBefore'],df_summary['happinessAfter'],alternative='greater')
# Print results
mean_pre = np.mean(df_summary['happinessBefore'].values)*100
mean_post = np.mean(df_summary['happinessAfter'].values)*100
n = df_summary.shape[0]
ste_pre = np.std(df_summary['happinessBefore'].values)*100/np.sqrt(n)
ste_post = np.std(df_summary['happinessBefore'].values)*100/np.sqrt(n)
CI = stats.norm.interval(alpha=0.95, loc=mean_post-mean_pre, scale=np.sqrt(ste_post**2 + ste_pre**2))

break_minutes = 7 # nominal duration of break period
print(f'Mean pre-break mood: {mean_pre:.03g}%, post_break mood: {mean_post:.03g}%, change in mood: {mean_post-mean_pre:.03g}% ({(mean_post-mean_pre)/break_minutes:.03g}%/min)')
print(f'95\%CI = ({CI[0]:.3g},{CI[1]:.3g})')
print(f'happinessBeforeActivities < happinessAfterActivities (PAIRED): T={t_less:.03g}, p={p_less:.03g}')
print(f'happinessBeforeActivities > happinessAfterActivities (PAIRED): T={t_more:.03g}, p={p_more:.03g}')
# Print conclusions
if p_less<0.05:
    print(f'** Free time break leads to DECREASED mood ratings in block {block_to_check}.')
elif p_more<0.05:
    print(f'** Free time break leads to INCREASED mood ratings in block {block_to_check}.')
else:
    print(f'** Free time break DOES NOT change mood ratings in block {block_to_check}.')


# %% Hyp 3.2: Compare effect of activities vs. rest period on mood
print('=======================================')
print('')
print('=======================================')
print("""
3.2) We hypothesize that the decrease in mood ratings will be less than that
    observed in the boredom task.
    We will test this with an unpaired sample t-test between (a) the difference
    in mood ratings before and after the break in the real-world task cohort,
    and (b) the difference in mood ratings at the start and end of the rest
    period in the boredom task cohort (both After-only and Before-and-after groups).
""")

batch_act = 'Activities'
# If repeat administration changes results, we'll use the after-only group.
# Otherwise, use both.
if use_both_boredom:
    batch_bored = 'AllBoredom'
else:
    batch_bored = 'BoredomAfterOnly'

print(f'=== POTD {batch_act} vs. {batch_bored} ===')
# Load mood ratings from Activities group
in_file = f'{results_dir}/Mmi-{batch_act}_Ratings.csv'
df_act = pd.read_csv(in_file)
# Create summary table of happiness scores
participants = np.unique(df_act.participant)
df_summary_act = pd.DataFrame(columns=['participant','happinessBefore','happinessAfter'])
df_summary_act['participant'] = participants
# Fill in with before-and-after happiness ratings
block_to_check = 0
for participant_index,participant in enumerate(participants):
    # extract happiness rating before and after break period
    df_this = df_act.loc[(df_act.participant==participant) & (df_act.iBlock==block_to_check),:]
    df_summary_act.loc[participant_index,'deltaHappiness'] = df_this.rating.values[-1] - df_this.rating.values[0]
    df_summary_act.loc[participant_index,'happinessBefore'] = df_this.rating.values[0]
    df_summary_act.loc[participant_index,'happinessAfter'] = df_this.rating.values[-1]

# === Do the same for boredom group
# Load mood ratings from boredom group
in_file = f'{results_dir}/Mmi-{batch_bored}_Ratings.csv'
df_boredom = pd.read_csv(in_file)
# Create summary table of happiness scores
participants = np.unique(df_boredom.participant)
df_summary_bored = pd.DataFrame(columns=['participant','happinessBefore','happinessAfter'])
df_summary_bored['participant'] = participants
# Fill in with before-and-after happiness ratings
for participant_index,participant in enumerate(participants):
    # extract happiness rating before and after rest period
    df_this = df_boredom.loc[(df_boredom.participant==participant) & (df_boredom.iBlock==block_to_check),:]
    df_summary_bored.loc[participant_index,'deltaHappiness'] = df_this.rating.values[-1] - df_this.rating.values[0]
    df_summary_bored.loc[participant_index,'happinessBefore'] = df_this.rating.values[0]
    df_summary_bored.loc[participant_index,'happinessAfter'] = df_this.rating.values[-1]


# Test stats with 2 one-sided t-tests
mean_act = np.mean(df_summary_act['deltaHappiness'].values)*100
mean_bored = np.mean(df_summary_bored['deltaHappiness'].values)*100
n_act = df_summary_act.shape[0]
n_bored = df_summary_bored.shape[0]
ste_act = np.std(df_summary_act['deltaHappiness'].values)*100/np.sqrt(n_act)
ste_bored = np.std(df_summary_bored['deltaHappiness'].values)*100/np.sqrt(n_bored)
CI = stats.norm.interval(alpha=0.95, loc=mean_act-mean_bored, scale=np.sqrt(ste_act**2 + ste_bored**2))

break_minutes = 7 # nominal duration of break period
print(f'Mean activities: {mean_act:.03g}%, Mean boredom: {mean_bored:.03g}')
print(f'95\%CI = ({CI[0]:.3g},{CI[1]:.3g})')


# Print results
print(f'activities < boredom: T={t_less:.03g}, p={p_less:.03g}')
print(f'activities > boredom: T={t_more:.03g}, p={p_more:.03g}')
# Print conclusions
if p_less<0.05:
    print(f'** Free time break happiness change is LESS than boredom happiness change in block {block_to_check}.')
elif p_more<0.05:
    print(f'** Free time break happiness change is GREATER than boredom happiness change in block {block_to_check}.')
else:
    print(f'** Free time break DOES NOT change happiness ratings more or less than boredom condition in block {block_to_check}.')

# %% Make a figure for this

# get means/stes of activities cohort
act_count = df_summary_act.shape[0]
act_mean_before = np.mean(df_summary_act['happinessBefore'])
act_ste_before = np.std(df_summary_act['happinessBefore'])/np.sqrt(act_count)
act_mean_after = np.mean(df_summary_act['happinessAfter'])
act_ste_after = np.std(df_summary_act['happinessAfter'])/np.sqrt(act_count)

# get means/stes of boredom cohort
bored_count = df_summary_bored.shape[0]
bored_mean_before = np.mean(df_summary_bored['happinessBefore'])
bored_ste_before = np.std(df_summary_bored['happinessBefore'])/np.sqrt(bored_count)
bored_mean_after = np.mean(df_summary_bored['happinessAfter'])
bored_ste_after = np.std(df_summary_bored['happinessAfter'])/np.sqrt(bored_count)

# plot
plt.figure(345,figsize=[9,6],clear=True)
plt.bar(np.arange(2)-0.2,[act_mean_before,act_mean_after],width=0.35,label=f'Free activities (n={act_count})')
plt.bar(np.arange(2)+0.2,[bored_mean_before,bored_mean_after],width=0.35,label=f'Rest + mood ratings (n={bored_count})')
mean_init_mood = np.mean([act_mean_before,bored_mean_before])
plt.axhline(mean_init_mood,c='k',ls='--',label='mean initial mood')
plt.errorbar(np.arange(2)-0.2,[act_mean_before,act_mean_after],[act_ste_before,act_ste_after],fmt='k.')
plt.errorbar(np.arange(2)+0.2,[bored_mean_before,bored_mean_after],[bored_ste_before,bored_ste_after],fmt='k.')
plt.legend()
plt.xticks(np.arange(2),['before','after'])
plt.ylabel('Mood (0-1)')
plt.grid()
plt.ylim(0.48, 0.72)
plt.title(f'Cohort {batch_act} vs. {batch_bored}')
plt.tight_layout()
plt.savefig(f'{figures_dir}/{batch_act}_vs_{batch_bored}_MoodBars.png')
plt.savefig(f'{figures_dir}/{batch_act}_vs_{batch_bored}_MoodBars.pdf')

# %% Get/print info about activities
batch = 'Activities'
in_file = f'{results_dir}/Mmi-{batch}_Probes.csv'
df_did = pd.read_csv(in_file)

# long version
questions = ['I thought.',
'I consumed the news.',
'I looked at photos.',
'I listened to music, podcasts, or radio.',
'I did some work for my (non-MTurk) job.',
'I looked for a (non-MTurk) job.',
'I paid bills, banked, or invested.',
'I did something else on my computer or phone.',
'I read texts or emails.',
'I wrote something.',
'I watched videos.',
'I went on social media.',
'I shopped.',
'I did something on MTurk.',
'I called/videochatted with someone.',
'I played a computer/phone game.',
'I did something on my computer/phone not listed here.',
'I read something NOT on a computer/phone.',
'I wrote something NOT on a computer/phone.',
'I watched TV.',
'I ate or drank something.',
'I spoke with someone in person.',
'I did a craft.',
'I stood up.',
'I did something physically active.',
'I went to the restroom.',
'I did something OFF a computer/phone not listed here.']
# short version
questions = ['thought','news','photos','audio','work','job-search','finances','other-comp-pre','messages','wrote','videos','social-media','shopped','mturk','called','game','other-comp','read','wrote','tv','ate','spoke','craft','stood','active','restroom','other-no-comp']
question_count = len(questions)
values = np.reshape(df_did.rating.values - 1, [-1,question_count]).T

plt.figure(14,figsize=[12,16],dpi=200,clear=True)
row_count = np.ceil(np.sqrt(question_count))
col_count = np.ceil(question_count/row_count)

freq_labels = ["Not at all", "A little", "About half the time", "A lot", "The whole time"]
freq_count = len(freq_labels)
bins = np.arange(freq_count+1)-0.5
act_hist = np.zeros([question_count,freq_count])
act_mean = np.zeros(question_count)
for question_index in range(question_count):
    act_hist[question_index] = np.histogram(values[question_index],bins)[0]
    act_mean[question_index] = np.mean(values[question_index])

# plot histo
plt.imshow(act_hist.T)
plt.xticks(np.arange(question_count),questions,rotation=45,ha='right')
plt.yticks(np.arange(freq_count),freq_labels)
plt.tight_layout()
plt.savefig(f'{figures_dir}/ActivitiesHisto.png')
plt.savefig(f'{figures_dir}/ActivitiesHisto.pdf')

# print in order of appearance
print('=======================================')
print('')
print('=======================================')
print('Frequency of activities reported (in order of appearance)')
for question_index,question in enumerate(questions):
    print(f'{question_index+1}. & {question} & {act_mean[question_index]/4*100:.03g}\% \\\ ')



# print in descending order
order = np.argsort(-act_mean)
print('=======================================')
print('')
print('=======================================')
print('Frequency of activities reported (in descending order)')
for order_index,question_index in enumerate(order):
    print(f'{order_index+1}. & {questions[question_index]} & {act_mean[question_index]/4*100:.03g}\% \\\ ')




# # %% Hyp 4.1.3: Floor effects in Effect of deltaBoredom on mood
# print('=======================================')
# print('')
# print('=======================================')
# print("""
# 4.0) Additionally, for hypotheses 1.2, 1.3, 1.4, 2.2, 2.3, and 2.4, if they
#     are significant, we will repeat the analyses including only mood points
#     collected prior to each participant's minimum mood value. If the results
#     are not significant in that case, we cannot rule out the possibility that
#     the effects we are observing are due to participants reaching their
#     minimum mood (floor effects).
# """)



# # %% Hyp 4.1.2: Floor effects in Effect of finalBoredom on mood

# print('=======================================')
# print('')
# print('=======================================')
# print("""
# 4.1.2) We hypothesize that final state boredom will explain variance in
#     subject-level POTD slope. This is a one-sided hypothesis.
#     We will test this with an ANOVA comparing the following two mixed effects
#     models (difference highlighted in bold):
#     H0: Mood ~ 1 + Time * (isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)
#     H1: Mood ~ 1 + Time * (finalBoredom + isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)
# """)
# # If repeat administration changes results, we'll use the after-only group.
# # Otherwise, use both.
# if use_both_boredom:
#     batch = 'AllBoredom'
# else:
#     batch = 'BoredomAfterOnly'

# print(f'=== Batch {batch}: Comparing LME models with and without finalBoredom ===')
# # Load pymer input file
# in_file = f'{results_dir}/Mmi-{batch}_pymerInput-full.csv'
# print(f'Opening {in_file}...')
# pymer_input = pd.read_csv(in_file, index_col=0)

# # Crop to exclude all mood ratings after miniumum rating (for floor effects)
# pymer_input = CropToMinRating(pymer_input)

# # Add finalBoredom scores
# in_file = f'{results_dir}/Mmi-{batch}_Probes.csv'
# print(f'Opening {in_file}...')
# df_boredom = pd.read_csv(in_file)
# participants = np.unique(df_boredom.participant)
# final_boredom = np.zeros(len(participants))
# for participant_index,participant in enumerate(participants):
#     # crop to this participant
#     df_this = df_boredom.loc[df_boredom.participant==participant,:]
#     # get final boredom score
#     final_boredom[participant_index] = np.sum(df_this.loc[df_this.iBlock==0,'rating']) # after first block
#     # add to pymer_input table
#     pymer_input.loc[pymer_input.Subject==participant,'finalBoredom'] = final_boredom[participant_index]

# # Fit models and run ANOVA to compare
# lm_string_h0 = 'Mood ~ 1 + Time * (isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)'
# lm_string_h1 = 'Mood ~ 1 + Time * (finalBoredom + isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)'
# anova_res, dfFit_h0, dfFit_h1, dfFixef_h0 = compare_lmers(pymer_input,lm_string_h0,lm_string_h1)
# print('ANOVA:')
# print(anova_res)

# # correlate new factor with subject LME slopes in reduced model
# PrintFactorSlopeCorrelations(dfFixef_h0,final_boredom,'finalBoredom',batch)

# # Print results and pymer fit
# if anova_res.loc[1,'Pr(>Chisq)']<0.05:
#     print('** (PRE-MIN ONLY) Final state boredom DOES explain added variance in subject-level POTD slope.')
# else:
#     print('** (PRE-MIN ONLY) Final state boredom does NOT explain added variance in subject-level POTD slope.')
# print(dfFit_h1.loc[['Time','finalBoredom','Time:finalBoredom']])

# # Print effect that a change of 1std would have on mood slope
# PrintEffectOf1StdChange(pymer_input,dfFit_h1,'finalBoredom')


# # %% Hyp 4.1.3: Floor effects in Effect of deltaBoredom on mood

# print('=======================================')
# print('')
# print('=======================================')
# print("""
# 4.1.3) We hypothesize that the change in boredom will explain variance in
#     subject-level POTD slope. This is a one-sided hypothesis.
#     We will test this with an ANOVA comparing the following two mixed effects
#     models (difference highlighted in bold):
#     H0: Mood ~ 1 + Time * (isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)
#     H1: Mood ~ 1 + Time * (deltaBoredom + isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)
#     If we fail to reject the null for hypothesis 1.1 (absolute cohen’s d is less
#     than 0.5) we will have to interpret the results of this hypothesis with the
#     caveat that it is possible that repeated administration of the state
#     boredom measure may have altered the results of the subsequent administration.
# """)

# # Analyzing change in boredom requires before-and-after group
# batch = 'BoredomBeforeAndAfter'

# print(f'=== Batch {batch}: Comparing LME models with and without deltaBoredom ===')
# # Load pymer input table
# in_file = f'{results_dir}/Mmi-{batch}_pymerInput-full.csv'
# print(f'Opening {in_file}...')
# pymer_input = pd.read_csv(in_file, index_col=0)

# # Crop to exclude all mood ratings after miniumum rating (for floor effects)
# pymer_input = CropToMinRating(pymer_input)

# # Add deltaBoredom scores
# in_file = f'{results_dir}/Mmi-{batch}_Probes.csv'
# print(f'Opening {in_file}...')
# df_boredom = pd.read_csv(in_file)
# participants = np.unique(df_boredom.participant)
# delta_boredom = np.zeros(len(participants))
# for participant_index,participant in enumerate(participants):
#     # crop to this participant
#     df_this = df_boredom.loc[df_boredom.participant==participant,:]
#     # get change in boredom scores
#     initial_boredom = np.sum(df_this.loc[df_this.iBlock==-1,'rating']) # after first block
#     final_boredom = np.sum(df_this.loc[df_this.iBlock==0,'rating']) # after first block
#     # add to pymer_input table
#     delta_boredom[participant_index] = final_boredom - initial_boredom
#     pymer_input.loc[pymer_input.Subject==participant,'deltaBoredom'] = delta_boredom[participant_index]

# # Fit models and run ANOVA to compare
# lm_string_h0 = 'Mood ~ 1 + Time * (isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)'
# lm_string_h1 = 'Mood ~ 1 + Time * (deltaBoredom + isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)'
# anova_res, dfFit_h0, dfFit_h1, dfFixef_h0 = compare_lmers(pymer_input,lm_string_h0,lm_string_h1)
# print('ANOVA:')
# print(anova_res)

# # correlate new factor with subject LME slopes in reduced model
# PrintFactorSlopeCorrelations(dfFixef_h0,delta_boredom,'deltaBoredom',batch)

# # Print results and fit
# if anova_res.loc[1,'Pr(>Chisq)']<0.05:
#     print('** (PRE-MIN ONLY) Change in state boredom DOES explain added variance in subject-level POTD slope.')
# else:
#     print('** (PRE-MIN ONLY) Change in state boredom does NOT explain added variance in subject-level POTD slope.')
# print(dfFit_h1.loc[['Time','deltaBoredom','Time:deltaBoredom']])

# # Print effect that a change of 1std would have on mood slope
# PrintEffectOf1StdChange(pymer_input,dfFit_h1,'deltaBoredom')

# # %% Hyp 4.1.4: Effect of traitBoredom on mood
# print('=======================================')
# print('')
# print('=======================================')
# print("""
# 4.1.4) We hypothesize that trait boredom will explain variance in subject-level
#     POTD slope.This is a one-sided hypothesis.
#     We will test this with an ANOVA comparing the following two mixed effects
#     models (difference highlighted in bold):
#     H0: Mood ~ 1 + Time * (isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)
#     H1: Mood ~ 1 + Time * (traitBoredom + isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)
# """)


# # If repeat administration changes results, we'll use the after-only group.
# # Otherwise, use both.
# if use_both_boredom:
#     batch = 'AllBoredom'
# else:
#     batch = 'BoredomAfterOnly'

# print(f'=== Batch {batch}: Comparing LME models with and without traitBoredom ===')
# # Load pymer input file
# in_file = f'{results_dir}/Mmi-{batch}_pymerInput-full.csv'
# print(f'Opening {in_file}...')
# pymer_input = pd.read_csv(in_file, index_col=0)

# # Crop to exclude all mood ratings after miniumum rating (for floor effects)
# pymer_input = CropToMinRating(pymer_input)

# # Add traitBoredom scores
# in_file = f'{results_dir}/Mmi-{batch}_Survey.csv'
# print(f'Opening {in_file}...')
# df_boredom = pd.read_csv(in_file)
# participants = np.unique(df_boredom.participant)
# trait_boredom = np.zeros(len(participants))
# for participant_index,participant in enumerate(participants):
#     trait_boredom[participant_index] = df_boredom.loc[df_boredom.participant==participant,'BORED'].values[0]
#     pymer_input.loc[pymer_input.Subject==participant,'traitBoredom'] = trait_boredom[participant_index]

# # Fit models and run ANOVA to compare
# lm_string_h0 = 'Mood ~ 1 + Time * (isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)'
# lm_string_h1 = 'Mood ~ 1 + Time * (traitBoredom + isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)'
# anova_res, dfFit_h0, dfFit_h1, dfFixef_h0 = compare_lmers(pymer_input,lm_string_h0,lm_string_h1)
# print('ANOVA:')
# print(anova_res)

# # correlate new factor with subject LME slopes in reduced model
# PrintFactorSlopeCorrelations(dfFixef_h0,trait_boredom,'traitBoredom',batch)

# # Print results and pymer fit
# if anova_res.loc[1,'Pr(>Chisq)']<0.05:
#     print('** (PRE-MIN ONLY) Trait boredom DOES explain added variance in subject-level POTD slope.')
# else:
#     print('** (PRE-MIN ONLY) Trait boredom does NOT explain added variance in subject-level POTD slope.')
# print(dfFit_h1.loc[['Time','traitBoredom','Time:traitBoredom']])

# # Print effect that a change of 1std would have on mood slope
# PrintEffectOf1StdChange(pymer_input,dfFit_h1,'traitBoredom')


# # %% Hyp 4.2.2: Floor effects in Effect of finalEmoDim on mood
# print('=======================================')
# print('')
# print('=======================================')
# print("""
# 4.2.2) We hypothesize that the final emotion dimension score will explain
#     variance in subject-level POTD slope. This is a one-sided hypothesis.
#     We will test this with an ANOVA comparing the following two mixed effects
#     models (difference highlighted in bold):
#     H0: Mood ~ 1 + Time * (isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)
#     H1: Mood ~ 1 + Time * (finalEmoDim + isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)
# """)

# # If repeat administration changes results, we'll use the after-only group.
# # Otherwise, use both.
# if use_both_mw:
#     batch = 'AllMw'
# else:
#     batch = 'MwAfterOnly'

# print(f'=== Batch {batch}: Comparing LME models with and without finalEmoDim ===')
# in_file = f'{results_dir}/Mmi-{batch}_pymerInput-full.csv'
# print(f'Opening {in_file}...')
# pymer_input = pd.read_csv(in_file, index_col=0)

# # Crop to exclude all mood ratings after miniumum rating (for floor effects)
# pymer_input = CropToMinRating(pymer_input)

# # Add finalEmoDim scores
# in_file = f'{results_dir}/Mmi-{batch}_Probes.csv'
# print(f'Opening {in_file}...')
# df_mw = pd.read_csv(in_file)
# participants = np.unique(df_mw.participant)
# final_mw = np.zeros(len(participants))
# for participant_index,participant in enumerate(participants):
#     # crop to this participant
#     df_this = df_mw.loc[df_mw.participant==participant,:]
#     # get final MW score
#     X_this = np.atleast_2d(df_this.loc[df_this.iBlock==0,'rating'])-0.5 # iBlock==0: after first block. -0.5: Move center of scale to 0
#     final_mw[participant_index] = pca.transform(X_this)[0,emotion_pc_index]
#     # Add to pymer_input table
#     pymer_input.loc[pymer_input.Subject==participant,'finalEmoDim'] = final_mw[participant_index]

# # Fit models and run ANOVA to compare
# lm_string_h0 = 'Mood ~ 1 + Time * (isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)'
# lm_string_h1 = 'Mood ~ 1 + Time * (finalEmoDim + isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)'
# anova_res, dfFit_h0, dfFit_h1, dfFixef_h0 = compare_lmers(pymer_input,lm_string_h0,lm_string_h1)
# print('ANOVA:')
# print(anova_res)

# # correlate new factor with subject LME slopes in reduced model
# PrintFactorSlopeCorrelations(dfFixef_h0,final_mw,'finalEmoDim',batch)

# # Print results and fit
# if anova_res.loc[1,'Pr(>Chisq)']<0.05:
#     print('** (PRE-MIN ONLY) Final MW emotion DOES explain added variance in subject-level POTD slope.')
# else:
#     print('** (PRE-MIN ONLY) Final MW emotion does NOT explain added variance in subject-level POTD slope.')
# print(dfFit_h1.loc[['Time','finalEmoDim','Time:finalEmoDim']])

# # Print effect that a change of 1std would have on mood slope
# PrintEffectOf1StdChange(pymer_input,dfFit_h1,'finalEmoDim')


# # %% Hyp 4.2.3: Floor effects in Effect of deltaEmoDim on mood
# print('=======================================')
# print('')
# print('=======================================')
# print("""
# 4.2.3) We hypothesize that the change in emotion dimension score will explain
#     variance in subject-level POTD slope. This is a one-sided hypothesis.
#     We will test this with an ANOVA comparing the following two mixed effects
#     models (difference highlighted in bold):
#     H0: Mood ~ 1 + Time * (isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)
#     H1: Mood ~ 1 + Time * (deltaEmoDim + isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)
#     If we fail to reject the null for hypothesis 2.1 (absolute cohen’s d is
#     less than 0.5) we will have to interpret the results of this hypothesis
#     with the caveat that it is possible that repeated administration of the
#     MDES measure may have altered the results of the subsequent administration.
# """)

# # Analyzing change in MW requires before-and-after group
# batch = 'MwBeforeAndAfter'

# print(f'=== Batch {batch}: Comparing LME models with and without deltaEmoDim ===')
# in_file = f'{results_dir}/Mmi-{batch}_pymerInput-full.csv'
# print(f'Opening {in_file}...')
# pymer_input = pd.read_csv(in_file, index_col=0)

# # Crop to exclude all mood ratings after miniumum rating (for floor effects)
# pymer_input = CropToMinRating(pymer_input)

# # Add deltaEmoDim scores
# in_file = f'{results_dir}/Mmi-{batch}_Probes.csv'
# print(f'Opening {in_file}...')
# df_mw = pd.read_csv(in_file)
# participants = np.unique(df_mw.participant)
# delta_mw = np.zeros(len(participants))
# for participant_index,participant in enumerate(participants):
#     # crop to this participant
#     df_this = df_mw.loc[df_mw.participant==participant,:]
#     # get initial MW score
#     X_initial = np.atleast_2d(df_this.loc[df_this.iBlock==-1,'rating'])-0.5 # Before 1st block. Move center of scale to 0
#     initial_mw = pca.transform(X_initial)[0,emotion_pc_index]
#     # get final MW score
#     X_final = np.atleast_2d(df_this.loc[df_this.iBlock==0,'rating'])-0.5 # After 1st block. Move center of scale to 0
#     final_mw = pca.transform(X_final)[0,emotion_pc_index]
#     # Add to pymer_input table
#     delta_mw[participant_index] = final_mw - initial_mw
#     pymer_input.loc[pymer_input.Subject==participant,'deltaEmoDim'] = delta_mw[participant_index]

# # Fit models and run ANOVA to compare
# lm_string_h0 = 'Mood ~ 1 + Time * (isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)'
# lm_string_h1 = 'Mood ~ 1 + Time * (deltaEmoDim + isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)'
# anova_res, dfFit_h0, dfFit_h1, dfFixef_h0 = compare_lmers(pymer_input,lm_string_h0,lm_string_h1)
# print('ANOVA:')
# print(anova_res)

# # correlate new factor with subject LME slopes in reduced model
# PrintFactorSlopeCorrelations(dfFixef_h0,delta_mw,'deltaEmoDim',batch)

# # Print results and fit
# if anova_res.loc[1,'Pr(>Chisq)']<0.05:
#     print('** (PRE-MIN ONLY) Change in MW emotion DOES explain added variance in subject-level POTD slope.')
# else:
#     print('** (PRE-MIN ONLY) Change in MW emotion does NOT explain added variance in subject-level POTD slope.')
# print(dfFit_h1.loc[['Time','deltaEmoDim','Time:deltaEmoDim']])

# # Print effect that a change of 1std would have on mood slope
# PrintEffectOf1StdChange(pymer_input,dfFit_h1,'deltaEmoDim')
