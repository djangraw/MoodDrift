#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TEMP_TestControlHypotheses.py

Test the MW control hypotheses in response to NHB reviewers.

Created on Fri Aug 19 11:16:43 2022

@author: djangraw
"""

# Import packages
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import seaborn as sns
# Import pymer functions
import CompareTwoLmers as c2l

# Declare file locations
results_dir = '../Data/OutFiles'
figures_dir = '../Figures' # where figures should be saved
use_both_mw = True; # because repeated administration did not affect results (in the emo dimension)
use_both_boredom = False; # because repeated administration DID affect results

# %% Declare functions
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

# Get last-mood minus first-mood for each participant in a list
def GetDeltaMood(pymer_input,participants):
    # get last-mood minus first-mood
    delta_mood = np.zeros(len(participants))
    for participant_index, participant in enumerate(participants):
        # pull out 1st-vs-last mood
        mood = pymer_input.loc[pymer_input.Subject==participant,'Mood'].values
        delta_mood[participant_index] = mood[-1]-mood[0]
    return delta_mood


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

# %% Get MW principal components
print('=======================================')
print('')
print('=======================================')

# print("""
# Mind Wandering Hypotheses:
# 2.1) In the validation of short-interval MDES repeat administration, we
#     hypothesize that the effect of including an initial administration will
#     have an absolute effect size (cohen’s d) less than 0.5.
#     We will test this with two, one-sided t-tests (TOST).
# """)

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
# # Save figure
# fig_file = f'{figures_dir}/{batch}_MwPca_VarExplained.png'
# print(f'Saving figure as {fig_file}...')
# fig.savefig(fig_file)

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

# # Save figure
# fig_file = f'{figures_dir}/{batch}_MwPcaLoadings.png'
# print(f'Saving figure as {fig_file}...')
# fig.savefig(fig_file)

# Note the most emotion-related PC
# defined as the one with the largest magnitude loading on the emotion question
emotion_pc_index = np.argmax(np.abs(pca.components_[:,question_labels=='emotion'])) # 4
# print(f'PC #{emotion_pc_index} appears to be emotion component.')


# %% Hyp 2.2: Effect of finalMW on mood
print('=======================================')
print('')
print('=======================================')
# print("""
# 2.2) We hypothesize that the final MDES scores will explain
#     variance in subject-level POTD slope. This is a one-sided hypothesis.
#     We will test this with an ANOVA comparing the following two mixed effects
#     models (difference highlighted in bold):
#     H0: Mood ~ 1 + all_finalMwPCs + Time * (isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)
#     H1: Mood ~ 1 + Time * (all_finalMwPCs + isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)
# """)

# If repeat administration changes results, we'll use the after-only group.
# Otherwise, use both.
if use_both_mw:
    batch = 'AllMw'
else:
    batch = 'MwAfterOnly'

# print(f'=== Batch {batch}: Comparing LME models with and without finalMW ===')
print(f'=== Batch {batch}: Comparing LME models with and without Time:fracRiskScore ===')
in_file = f'{results_dir}/Mmi-{batch}_pymerInput-full.csv'
print(f'Opening {in_file}...')
pymer_input = pd.read_csv(in_file, index_col=0)

anova_res = None
for do_premin_only in [False]:#,True]:

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
        # final_mw[participant_index] = pca.transform(X_this)[0,emotion_pc_index]
        # Add to pymer_input table
        # pymer_input.loc[pymer_input.Subject==participant,'finalEmoDim'] = final_mw[participant_index]
        for pc_index in range(pc_count):
            final_mw = pca.transform(X_this)[0,pc_index]
            pymer_input.loc[pymer_input.Subject==participant,f'finalMwPC{pc_index}'] = final_mw

    # Plot stat vs. change in mood
    # delta_mood = GetDeltaMood(pymer_input,participants)
    # MakeJointPlot(final_mw,delta_mood,'final_mw','delta_mood',cohort_name=cohort_name)

    # make PC string
    mw_pc_string = 'finalMwPC0'
    for pc_index in range(1,pc_count):
        mw_pc_string += f' + finalMwPC{pc_index}'

    # Fit models and run ANOVA to compare
    # lm_string_h0 = 'Mood ~ 1 + Time * (isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)'
    # lm_string_h0 = f'Mood ~ 1 + Time : ({mw_pc_string}) + Time * (isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)'
    lm_string_h0 = f'Mood ~ 1 + {mw_pc_string} + Time * (isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)'
    lm_string_h1 = f'Mood ~ 1 + Time * ({mw_pc_string} + isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)'
    lm_string_null = 'Mood ~ 1 + (1 + Time|Subject)'

    # run anova and print results
    anova_res, dfFit_h0, dfFit_h1, dfFixef_h0 = c2l.compare_lmers(pymer_input,lm_string_h0,lm_string_h1,lm_string_null)
    c2l.print_comparison_results(batch,pymer_input,lm_string_h0,lm_string_h1,anova_res,dfFit_h0,dfFit_h1)

    # correlate new factor with subject LME slopes in reduced model
    # PrintFactorSlopeCorrelations(dfFixef_h0,final_mw,'finalEmoDim',cohort_name)

    # Print results and fit
    if anova_res.loc[1,'Pr(>Chisq)']<0.05:
        print('** Final MW content DOES explain added variance in subject-level POTD slope.')
    else:
        print('** Final MW content does NOT explain added variance in subject-level POTD slope.')

    # for pc_index in range(pc_count):
    #     PrintEffectOf1StdChange(pymer_input,dfFit_h1,f'finalMwPC{pc_index}')
    # Print effect that a change of 1std would have on mood slope
    # PrintEffectOf1StdChange(pymer_input,dfFit_h1,'finalEmoDim')




# %% Hyp 2.3: Effect of deltaMW on mood
print('=======================================')
print('')
print('=======================================')
# print("""
# 2.3) We hypothesize that the change in MDES scores will explain
#     variance in subject-level POTD slope. This is a one-sided hypothesis.
#     We will test this with an ANOVA comparing the following two mixed effects
#     models (difference highlighted in bold):
#     H0: Mood ~ 1 + all_deltaMwPCs + Time * (isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)
#     H1: Mood ~ 1 + Time * (all_deltaMwPCs + isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)
#     If we fail to reject the null for hypothesis 2.1 (absolute cohen’s d is
#     less than 0.5) we will have to interpret the results of this hypothesis
#     with the caveat that it is possible that repeated administration of the
#     MDES measure may have altered the results of the subsequent administration.
# """)

# Analyzing change requires before-and-after batch
batch = 'MwBeforeAndAfter'

# print(f'=== Batch {batch}: Comparing LME models with and without deltaEmoDim ===')
print(f'=== Batch {batch}: Comparing LME models with and without Time:fracRiskScore ===')
# Load pymyer input file
in_file = f'{results_dir}/Mmi-{batch}_pymerInput-full.csv'
print(f'Opening {in_file}...')
pymer_input = pd.read_csv(in_file, index_col=0)

anova_res = None
for do_premin_only in [False]:#,True]:

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
        # initial_mw[participant_index] = pca.transform(X_initial)[0,emotion_pc_index]
        # get final MW score
        X_final = np.atleast_2d(df_this.loc[df_this.iBlock==0,'rating'])-0.5 # After 1st block. Move center of scale to 0
        # final_mw[participant_index] = pca.transform(X_final)[0,emotion_pc_index]
        for pc_index in range(pc_count):
            initial_mw = pca.transform(X_initial)[0,pc_index]
            final_mw = pca.transform(X_final)[0,pc_index]
            delta_mw = final_mw - initial_mw
            pymer_input.loc[pymer_input.Subject==participant,f'deltaMwPC{pc_index}'] = delta_mw

        # Add to pymer_input table
        # delta_mw[participant_index] = final_mw[participant_index] - initial_mw[participant_index]
        # pymer_input.loc[pymer_input.Subject==participant,'deltaEmoDim'] = delta_mw[participant_index]


    # Plot stat vs. change in mood
    # delta_mood = GetDeltaMood(pymer_input,participants)
    # MakeJointPlot(delta_mw,delta_mood,'delta_mw','delta_mood',cohort_name=cohort_name)
    # # for completeness, also do initial & final for this group
    # MakeJointPlot(initial_mw,delta_mood,'initial_mw','delta_mood',cohort_name=cohort_name)
    # MakeJointPlot(final_mw,delta_mood,'final_mw','delta_mood',cohort_name=cohort_name)

    # make PC string
    mw_pc_string = 'deltaMwPC0'
    for pc_index in range(1,pc_count):
        mw_pc_string += f' + deltaMwPC{pc_index}'

    # Fit models and run ANOVA to compare
    # lm_string_h0 = 'Mood ~ 1 + Time * (isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)'
    # lm_string_h1 = f'Mood ~ 1 + Time * ({mw_pc_string} + isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)'
    lm_string_h0 = f'Mood ~ 1 + {mw_pc_string} + Time * (isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)'
    lm_string_h1 = f'Mood ~ 1 + Time * ({mw_pc_string} + isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)'
    lm_string_null = 'Mood ~ 1 + (1 + Time|Subject)'

    # run anova and print results
    anova_res, dfFit_h0, dfFit_h1, dfFixef_h0 = c2l.compare_lmers(pymer_input,lm_string_h0,lm_string_h1,lm_string_null)
    c2l.print_comparison_results(batch,pymer_input,lm_string_h0,lm_string_h1,anova_res,dfFit_h0,dfFit_h1)

    # correlate new factor with subject LME slopes in reduced model
    # PrintFactorSlopeCorrelations(dfFixef_h0,delta_mw,'deltaEmoDim',cohort_name)

    # Print results and fit
    if anova_res.loc[1,'Pr(>Chisq)']<0.05:
        print('** Change in MW content DOES explain added variance in subject-level POTD slope.')
    else:
        print('** Change in MW content does NOT explain added variance in subject-level POTD slope.')

    # for pc_index in range(pc_count):
    #     PrintEffectOf1StdChange(pymer_input,dfFit_h1,f'deltaMwPC{pc_index}')

    # Print effect that a change of 1std would have on mood slope
    # PrintEffectOf1StdChange(pymer_input,dfFit_h1,'deltaEmoDim')



# %% Hyp 2.4: Effect of traitMW on mood
print('=======================================')
print('')
print('=======================================')
# print("""
# 2.4) We hypothesize that trait mind wandering will explain variance in
#     subject-level POTD slope. This is a one-sided hypothesis.
#     We will test this with an ANOVA comparing the following two mixed effects
#     models (difference highlighted in bold):
#     H0: Mood ~ 1 + Time * (isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)
#     H1: Mood ~ 1 + Time * (traitMW + isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)
# """)

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
    # delta_mood = GetDeltaMood(pymer_input,participants)
    # MakeJointPlot(trait_mw,delta_mood,'trait_mw','delta_mood',cohort_name=cohort_name)

    # Fit models and run ANOVA to compare
    # lm_string_h0 = 'Mood ~ 1 + Time * (isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)'
    # lm_string_h1 = 'Mood ~ 1 + Time * (finalBoredom + isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)'
    lm_string_h0 = 'Mood ~ 1 + traitMW + Time * (isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)'
    lm_string_h1 = 'Mood ~ 1 + Time * (traitMW + isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)'
    lm_string_null = 'Mood ~ 1 + (1 + Time|Subject)'

    # run anova and print results
    anova_res, dfFit_h0, dfFit_h1, dfFixef_h0 = c2l.compare_lmers(pymer_input,lm_string_h0,lm_string_h1,lm_string_null)
    c2l.print_comparison_results(batch,pymer_input,lm_string_h0,lm_string_h1,anova_res,dfFit_h0,dfFit_h1)


    # correlate new factor with subject LME slopes in reduced model
    # PrintFactorSlopeCorrelations(dfFixef_h0,trait_mw,'traitMW',cohort_name)

    # Print results and fit
    if anova_res.loc[1,'Pr(>Chisq)']<0.05:
        print('** Trait MW DOES explain added variance in subject-level POTD slope.')
    else:
        print('** Trait MW does NOT explain added variance in subject-level POTD slope.')
    print(dfFit_h1.loc[['Time','traitMW','Time:traitMW']])

    # Print effect that a change of 1std would have on mood slope
    # PrintEffectOf1StdChange(pymer_input,dfFit_h1,'traitMW')






# %% Hyp 1.2: Effect of finalBoredom on mood
print('=======================================')
print('')
print('=======================================')
# print("""
# 1.2) We hypothesize that final state boredom will explain variance in
#     subject-level POTD slope. This is a one-sided hypothesis.
#     We will test this with an ANOVA comparing the following two mixed effects
#     models (difference highlighted in bold):
#     H0: Mood ~ 1 + finalBoredom + Time * (isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)
#     H1: Mood ~ 1 + Time * (finalBoredom + isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)
# """)


# If repeat administration changes results, we'll use the after-only group.
# Otherwise, use both.
if use_both_boredom:
    batch = 'AllBoredom'
else:
    batch = 'BoredomAfterOnly'

# print(f'=== Batch {batch}: Comparing LME models with and without finalBoredom ===')
print(f'=== Batch {batch}: Comparing LME models with and without Time:fracRiskScore ===')
# Load pymer input file
in_file = f'{results_dir}/Mmi-{batch}_pymerInput-full.csv'
print(f'Opening {in_file}...')
pymer_input = pd.read_csv(in_file, index_col=0)

anova_res = None
for do_premin_only in [False]:#,True]:

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
    # MakeJointPlot(final_boredom,delta_mood,'final_boredom','delta_mood',cohort_name=cohort_name)

    # Fit models and run ANOVA to compare
    # lm_string_h0 = 'Mood ~ 1 + Time * (isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)'
    # lm_string_h1 = 'Mood ~ 1 + Time * (finalBoredom + isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)'
    lm_string_h0 = 'Mood ~ 1 + finalBoredom + Time * (isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)'
    lm_string_h1 = 'Mood ~ 1 + Time * (finalBoredom + isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)'
    lm_string_null = 'Mood ~ 1 + (1 + Time|Subject)'

    # run anova and print results
    anova_res, dfFit_h0, dfFit_h1, dfFixef_h0 = c2l.compare_lmers(pymer_input,lm_string_h0,lm_string_h1,lm_string_null)
    c2l.print_comparison_results(batch,pymer_input,lm_string_h0,lm_string_h1,anova_res,dfFit_h0,dfFit_h1)

    # correlate new factor with subject LME slopes in reduced model
    # PrintFactorSlopeCorrelations(dfFixef_h0,final_boredom,'finalBoredom',cohort_name)

    # Print results and pymer fit
    if anova_res.loc[1,'Pr(>Chisq)']<0.05:
        print('** Final state boredom DOES explain added variance in subject-level POTD slope.')
    else:
        print('** Final state boredom does NOT explain added variance in subject-level POTD slope.')
    # print(dfFit_h1.loc[['Time','finalBoredom','Time:finalBoredom']])

    # Print effect that a change of 1std would have on mood slope
    # PrintEffectOf1StdChange(pymer_input,dfFit_h1,'finalBoredom')


# %% Hyp 1.3: Effect of deltaBoredom on mood
print('=======================================')
print('')
print('=======================================')
# print("""
# 1.3) We hypothesize that the change in boredom will explain variance in
#     subject-level POTD slope. This is a one-sided hypothesis.
#     We will test this with an ANOVA comparing the following two mixed effects
#     models (difference highlighted in bold):
#     H0: Mood ~ 1 + deltaBoredom + Time * (isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)
#     H1: Mood ~ 1 + Time * (deltaBoredom + isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)
#     If we fail to reject the null for hypothesis 1.1 (absolute cohen’s d is less
#     than 0.5) we will have to interpret the results of this hypothesis with the
#     caveat that it is possible that repeated administration of the state
#     boredom measure may have altered the results of the subsequent administration.
# """)

# Analyzing change in boredom requires before-and-after group
batch = 'BoredomBeforeAndAfter'

# print(f'=== Batch {batch}: Comparing LME models with and without deltaBoredom ===')
print(f'=== Batch {batch}: Comparing LME models with and without Time:fracRiskScore ===')
# Load pymer input file
in_file = f'{results_dir}/Mmi-{batch}_pymerInput-full.csv'
print(f'Opening {in_file}...')
pymer_input = pd.read_csv(in_file, index_col=0)

anova_res = None
for do_premin_only in [False]:#,True]:

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
    # MakeJointPlot(df_summary['delta_boredom'],delta_mood,'delta_boredom','delta_mood',cohort_name=cohort_name)
    # for completeness, also do initial & final for this group
    # MakeJointPlot(df_summary['initial_boredom'],delta_mood,'initial_boredom','delta_mood',cohort_name=cohort_name)
    # MakeJointPlot(df_summary['final_boredom'],delta_mood,'final_boredom','delta_mood',cohort_name=cohort_name)

    # Fit models and run ANOVA to compare
    # lm_string_h0 = 'Mood ~ 1 + Time * (isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)'
    # lm_string_h1 = 'Mood ~ 1 + Time * (deltaBoredom + isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)'
    lm_string_h0 = 'Mood ~ 1 + deltaBoredom + Time * (isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)'
    lm_string_h1 = 'Mood ~ 1 + Time * (deltaBoredom + isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)'
    lm_string_null = 'Mood ~ 1 + (1 + Time|Subject)'

    # run anova and print results
    anova_res, dfFit_h0, dfFit_h1, dfFixef_h0 = c2l.compare_lmers(pymer_input,lm_string_h0,lm_string_h1,lm_string_null)
    c2l.print_comparison_results(batch,pymer_input,lm_string_h0,lm_string_h1,anova_res,dfFit_h0,dfFit_h1)

    # correlate new factor with subject LME slopes in reduced model
    # PrintFactorSlopeCorrelations(dfFixef_h0,df_summary['delta_boredom'],'deltaBoredom',cohort_name)

    # Print results and pymer fit
    if anova_res.loc[1,'Pr(>Chisq)']<0.05:
        print('** Change in state boredom DOES explain added variance in subject-level POTD slope.')
    else:
        print('** Change in state boredom does NOT explain added variance in subject-level POTD slope.')
    # print(dfFit_h1.loc[['Time','deltaBoredom','Time:deltaBoredom']])

    # Print effect that a change of 1std would have on mood slope
    # PrintEffectOf1StdChange(pymer_input,dfFit_h1,'deltaBoredom')



# %% Hyp 1.4: Effect of traitBoredom on mood
print('=======================================')
print('')
print('=======================================')
# print("""
# 1.4) We hypothesize that trait boredom will explain variance in subject-level
#     POTD slope.This is a one-sided hypothesis.
#     We will test this with an ANOVA comparing the following two mixed effects
#     models (difference highlighted in bold):
#     H0: Mood ~ 1 + Time * (isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)
#     H1: Mood ~ 1 + Time * (traitBoredom + isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)
# """)


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
    # lm_string_h0 = 'Mood ~ 1 + Time * (isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)'
    # lm_string_h1 = 'Mood ~ 1 + Time * (deltaBoredom + isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)'
    lm_string_h0 = 'Mood ~ 1 + traitBoredom + Time * (isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)'
    lm_string_h1 = 'Mood ~ 1 + Time * (traitBoredom + isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)'
    lm_string_null = 'Mood ~ 1 + (1 + Time|Subject)'

    # run anova and print results
    anova_res, dfFit_h0, dfFit_h1, dfFixef_h0 = c2l.compare_lmers(pymer_input,lm_string_h0,lm_string_h1,lm_string_null)
    c2l.print_comparison_results(batch,pymer_input,lm_string_h0,lm_string_h1,anova_res,dfFit_h0,dfFit_h1)

    # correlate new factor with subject LME slopes in reduced model
    # PrintFactorSlopeCorrelations(dfFixef_h0,trait_boredom,'traitBoredom',cohort_name)

    # Print results and pymer fit
    if anova_res.loc[1,'Pr(>Chisq)']<0.05:
        print('** Trait boredom DOES explain added variance in subject-level POTD slope.')
    else:
        print('** Trait boredom does NOT explain added variance in subject-level POTD slope.')
    print(dfFit_h1.loc[['Time','traitBoredom','Time:traitBoredom']])

    # Print effect that a change of 1std would have on mood slope
    # PrintEffectOf1StdChange(pymer_input,dfFit_h1,'traitBoredom')
