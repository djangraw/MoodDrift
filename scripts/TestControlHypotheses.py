#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TestControlsHypotheses.py

Test the control analyses preregistered on osf.

Created on Thu Dec 16 15:27:29 2021
@author: djangraw
- Updated 12/22/21 by DJ - finished script, commented.
"""

# Import packages
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
# Import pymer functions
from pymer4.models import Lmer
from rpy2 import robjects

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

    return df_summary


# Declare constants
results_dir = '../Data/OutFiles'
batch_ba = 'BoredomBeforeAndAfter' # before-and-after group, got thought probes both before and after rest block
batch_ao = 'BoredomAfterOnly' # after-only group, got thought probes only after rest block
figures_folder = '../Figures' # where figures should be saved

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
    # Run 2 one-sided t-tests
    t_less,p_less = stats.ttest_ind(df_summary_ba[block_to_check],df_summary_ao[block_to_check],alternative='less')
    t_more,p_more = stats.ttest_ind(df_summary_ba[block_to_check],df_summary_ao[block_to_check],alternative='greater')
    # Print results
    print(f'BoredomBeforeAndAfter < BoredomAfterOnly: T={t_less:.03g}, p={p_less:.03g}')
    print(f'BoredomBeforeAndAfter > BoredomAfterOnly: T={t_more:.03g}, p={p_more:.03g}')
    # Print conclusions
    if p_less<0.05:
        print(f'** Presenting boredom questions before start of task leads to DECREASED responses after {block_to_check}.')
    elif p_more<0.05:
        print(f'** Presenting boredom questions before start of task leads to INCREASED responses after {block_to_check}.')
    else:
        print(f'** Presenting boredom questions before start of task DOES NOT change responses after {block_to_check}.')



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

# Function to fit 2 alternative LME models and compare the variance explained with an ANOVA 
def CompareModels(pymer_input,lm_string_h0,lm_string_h1):
    # fit model for H0
    model_h0 = Lmer(lm_string_h0, data=pymer_input)
    _ = model_h0.fit(old_optimizer=True)
    dfFit_h0 = model_h0.coefs
    # fit model for H1
    model_h1 = Lmer(lm_string_h1, data=pymer_input)
    _ = model_h1.fit(old_optimizer=True)
    dfFit_h1 = model_h1.coefs
    # print ANOVA results
    # robjects.r('print')(robjects.r('anova')(model_h0.model_obj, model_h1.model_obj)) # r version
    anova_res = pd.DataFrame((robjects.r('anova')(model_h0.model_obj, model_h1.model_obj))) # pandas version
    print(anova_res)
    # Return results
    return anova_res, dfFit_h0, dfFit_h1
    

# Since repeat administration changes results, we'll use the after-only group.
batch = 'BoredomAfterOnly'

print(f'=== Batch {batch}: Comparing LME models with and without finalBoredom ===')
# Load pymer input file
in_file = f'{results_dir}/Mmi-{batch}_pymerInput-full.csv'
print(f'Opening {in_file}...')
pymer_input = pd.read_csv(in_file, index_col=0)

# Add finalBoredom scores
in_file = f'{results_dir}/Mmi-{batch}_Probes.csv'
print(f'Opening {in_file}...')
df_boredom = pd.read_csv(in_file)
participants = np.unique(df_boredom.participant)
for participant_index,participant in enumerate(participants):
    df_this = df_boredom.loc[df_boredom.participant==participant,:]
    final_boredom = np.sum(df_this.loc[df_this.iBlock==0,'rating']) # after first block
    pymer_input.loc[pymer_input.Subject==participant,'finalBoredom'] = final_boredom

# Fit models and run ANOVA to compare
lm_string_h0 = 'Mood ~ 1 + Time * (isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)'
lm_string_h1 = 'Mood ~ 1 + Time * (finalBoredom + isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)'
anova_res, dfFit_h0, dfFit_h1 = CompareModels(pymer_input,lm_string_h0,lm_string_h1)

# Print results and pymer fit
if anova_res.loc[1,'Pr(>Chisq)']<0.05:
    print('** Final state boredom DOES explain added variance in subject-level POTD slope.')
else:
    print('** Final state boredom does NOT explain added variance in subject-level POTD slope.')
print(dfFit_h1.loc[['finalBoredom','Time:finalBoredom']])

    
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

# Add deltaBoredom scores
in_file = f'{results_dir}/Mmi-{batch}_Probes.csv'
print(f'Opening {in_file}...')
df_boredom = pd.read_csv(in_file)
participants = np.unique(df_boredom.participant)
for participant_index,participant in enumerate(participants):
    df_this = df_boredom.loc[df_boredom.participant==participant,:]
    initial_boredom = np.sum(df_this.loc[df_this.iBlock==-1,'rating']) # after first block
    final_boredom = np.sum(df_this.loc[df_this.iBlock==0,'rating']) # after first block
    pymer_input.loc[pymer_input.Subject==participant,'deltaBoredom'] = final_boredom - initial_boredom

# Fit models and run ANOVA to compare
lm_string_h0 = 'Mood ~ 1 + Time * (isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)'
lm_string_h1 = 'Mood ~ 1 + Time * (deltaBoredom + isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)'
anova_res, dfFit_h0, dfFit_h1 = CompareModels(pymer_input,lm_string_h0,lm_string_h1)

# Print results and pymer fit
if anova_res.loc[1,'Pr(>Chisq)']<0.05:
    print('** Change in state boredom DOES explain added variance in subject-level POTD slope.')
else:
    print('** Change in state boredom does NOT explain added variance in subject-level POTD slope.')
print(dfFit_h1.loc[['deltaBoredom','Time:deltaBoredom']])


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


# Since repeat administration changes results, we'll use the after-only group.
batch = 'BoredomAfterOnly'

print(f'=== Batch {batch}: Comparing LME models with and without traitBoredom ===')
# Load pymer input file
in_file = f'{results_dir}/Mmi-{batch}_pymerInput-full.csv'
print(f'Opening {in_file}...')
pymer_input = pd.read_csv(in_file, index_col=0)

# Add traitBoredom scores
in_file = f'{results_dir}/Mmi-{batch}_Survey.csv'
print(f'Opening {in_file}...')
df_boredom = pd.read_csv(in_file)
participants = np.unique(df_boredom.participant)
for participant_index,participant in enumerate(participants):
    boredom_score = df_boredom.loc[df_boredom.participant==participant,'BORED'].values[0]
    pymer_input.loc[pymer_input.Subject==participant,'traitBoredom'] = boredom_score

# Fit models and run ANOVA to compare
lm_string_h0 = 'Mood ~ 1 + Time * (isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)'
lm_string_h1 = 'Mood ~ 1 + Time * (traitBoredom + isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)'
anova_res, dfFit_h0, dfFit_h1 = CompareModels(pymer_input,lm_string_h0,lm_string_h1)

# Print results and pymer fit
if anova_res.loc[1,'Pr(>Chisq)']<0.05:
    print('** Trait boredom DOES explain added variance in subject-level POTD slope.')
else:
    print('** Trait boredom does NOT explain added variance in subject-level POTD slope.')
print(dfFit_h1.loc[['traitBoredom','Time:traitBoredom']])


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
fig_file = f'{figures_folder}/{batch}_MwPca_VarExplained.png'
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
fig_file = f'{figures_folder}/{batch}_MwPcaLoadings.png'
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
    print(f'=== {block_to_check} ===')
    t_less,p_less = stats.ttest_ind(df_summary_ba[block_to_check],df_summary_ao[block_to_check],alternative='less')
    t_more,p_more = stats.ttest_ind(df_summary_ba[block_to_check],df_summary_ao[block_to_check],alternative='greater')
    print(f'MwBeforeAndAfter < MwAfterOnly: T={t_less:.03g}, p={p_less:.03g}')
    print(f'MwBeforeAndAfter > MwAfterOnly: T={t_more:.03g}, p={p_more:.03g}')
    if p_less<0.05:
        print(f'** Presenting MW questions before start of task leads to DECREASED responses after {block_to_check}.')
    elif p_more<0.05:
        print(f'** Presenting MW questions before start of task leads to INCREASED responses after {block_to_check}.')
    else:
        print(f'** Presenting MW questions before start of task DOES NOT change responses after {block_to_check}.')


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

# Because 2.1 was not significant, use both before-and-after and after-only batches.
batch = 'AllMw'
# If 2.1 were not significant, we'd use after-only batch.
# batch = 'MwAfterOnly'

print(f'=== Batch {batch}: Comparing LME models with and without finalEmoDim ===')
in_file = f'{results_dir}/Mmi-{batch}_pymerInput-full.csv'
print(f'Opening {in_file}...')
pymer_input = pd.read_csv(in_file, index_col=0)

# Add finalEmoDim scores
in_file = f'{results_dir}/Mmi-{batch}_Probes.csv'
print(f'Opening {in_file}...')
df_mw = pd.read_csv(in_file)
participants = np.unique(df_mw.participant)
for participant_index,participant in enumerate(participants):
    # crop to this participant
    df_this = df_mw.loc[df_mw.participant==participant,:]
    # get final MW score
    X_this = np.atleast_2d(df_this.loc[df_this.iBlock==0,'rating'])-0.5 # iBlock==0: after first block. -0.5: Move center of scale to 0
    final_mw = pca.transform(X_this)[0,emotion_pc_index] 
    # Add to pymer_input table
    pymer_input.loc[pymer_input.Subject==participant,'finalEmoDim'] = final_mw

# Fit models and run ANOVA to compare
lm_string_h0 = 'Mood ~ 1 + Time * (isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)'
lm_string_h1 = 'Mood ~ 1 + Time * (finalEmoDim + isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)'
anova_res, dfFit_h0, dfFit_h1 = CompareModels(pymer_input,lm_string_h0,lm_string_h1)

# Print results and fit
if anova_res.loc[1,'Pr(>Chisq)']<0.05:
    print('** Final MW emotion DOES explain added variance in subject-level POTD slope.')
else:
    print('** Final MW emotion does NOT explain added variance in subject-level POTD slope.')
print(dfFit_h1.loc[['finalEmoDim','Time:finalEmoDim']])
    

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

# Add deltaEmoDim scores
in_file = f'{results_dir}/Mmi-{batch}_Probes.csv'
print(f'Opening {in_file}...')
df_mw = pd.read_csv(in_file)
participants = np.unique(df_mw.participant)
for participant_index,participant in enumerate(participants):
    # crop to this participant
    df_this = df_mw.loc[df_mw.participant==participant,:]
    # get initial MW score
    X_initial = np.atleast_2d(df_this.loc[df_this.iBlock==-1,'rating'])-0.5 # Before 1st block. Move center of scale to 0
    initial_mw = pca.transform(X_initial)[0,emotion_pc_index] 
    # get final MW score
    X_final = np.atleast_2d(df_this.loc[df_this.iBlock==0,'rating'])-0.5 # After 1st block. Move center of scale to 0
    final_mw = pca.transform(X_final)[0,emotion_pc_index] 
    # Add to pymer_input table
    pymer_input.loc[pymer_input.Subject==participant,'deltaEmoDim'] = final_mw - initial_mw

# Fit models and run ANOVA to compare
lm_string_h0 = 'Mood ~ 1 + Time * (isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)'
lm_string_h1 = 'Mood ~ 1 + Time * (deltaEmoDim + isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)'
anova_res, dfFit_h0, dfFit_h1 = CompareModels(pymer_input,lm_string_h0,lm_string_h1)

# Print results and fit
if anova_res.loc[1,'Pr(>Chisq)']<0.05:
    print('** Change in MW emotion DOES explain added variance in subject-level POTD slope.')
else:
    print('** Change in MW emotion does NOT explain added variance in subject-level POTD slope.')
print(dfFit_h1.loc[['deltaEmoDim','Time:deltaEmoDim']])


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

# Because 2.1 showed no effect of repeat administration, we can use both groups
batch = 'AllMw'

print(f'=== Batch {batch}: Comparing LME models with and without traitMW ===')
# load pymer input tables
in_file = f'{results_dir}/Mmi-{batch}_pymerInput-full.csv'
print(f'Opening {in_file}...')
pymer_input = pd.read_csv(in_file, index_col=0)

# Add traitMW scores
in_file = f'{results_dir}/Mmi-{batch}_Survey.csv'
print(f'Opening {in_file}...')
df_mw = pd.read_csv(in_file)
participants = np.unique(df_mw.participant)
for participant_index,participant in enumerate(participants):
    # get trait MW score from table
    mw_score = df_mw.loc[df_mw.participant==participant,'MW'].values[0]
    # add to pymer_input table
    pymer_input.loc[pymer_input.Subject==participant,'traitMW'] = mw_score

# Fit models and run ANOVA to compare
lm_string_h0 = 'Mood ~ 1 + Time * (isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)'
lm_string_h1 = 'Mood ~ 1 + Time * (traitMW + isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)'
anova_res, dfFit_h0, dfFit_h1 = CompareModels(pymer_input,lm_string_h0,lm_string_h1)

# Print results and fit
if anova_res.loc[1,'Pr(>Chisq)']<0.05:
    print('** Trait MW DOES explain added variance in subject-level POTD slope.')
else:
    print('** Trait MW does NOT explain added variance in subject-level POTD slope.')
print(dfFit_h1.loc[['traitMW','Time:traitMW']])


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
print(f'happinessBeforeActivities < happinessAfterActivities (PAIRED): T={t_less:.03g}, p={p_less:.03g}')
print(f'happinessBeforeActivities > happinessAfterActivities (PAIRED): T={t_more:.03g}, p={p_more:.03g}')
# Print conclusions
if p_less<0.05:
    print(f'** Free time break leads to DECREASED responses in block {block_to_check}.')
elif p_more<0.05:
    print(f'** Free time break leads to INCREASED responses in block {block_to_check}.')
else:
    print(f'** Free time break DOES NOT change responses in block {block_to_check}.')


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
batch_bored = 'BoredomAfterOnly' # because repeat administration affected data, use after-only group.

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
    

# Test stats with 2 one-sided t-tests     
t_less,p_less = stats.ttest_ind(df_summary_act['deltaHappiness'],df_summary_bored['deltaHappiness'],alternative='less')
t_more,p_more = stats.ttest_ind(df_summary_act['deltaHappiness'],df_summary_bored['deltaHappiness'],alternative='greater')
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

# %% Hyp 4.1.3: Floor effects in Effect of deltaBoredom on mood
print('=======================================')
print('')
print('=======================================')
print("""
4.0) Additionally, for hypotheses 1.2, 1.3, 1.4, 2.2, 2.3, and 2.4, if they  
    are significant, we will repeat the analyses including only mood points 
    collected prior to each participant's minimum mood value. If the results 
    are not significant in that case, we cannot rule out the possibility that 
    the effects we are observing are due to participants reaching their 
    minimum mood (floor effects).
""")

print('=======================================')
print('')
print('=======================================')
print("""
4.1.3) We hypothesize that the change in boredom will explain variance in  
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

# Crop to exclude all mood ratings after miniumum rating (for floor effects)
def CropToMinRating(pymer_input):
    participants = np.unique(pymer_input.Subject)
    for participant_index,participant in enumerate(participants):
        df_this = pymer_input.loc[pymer_input.Subject==participant,:]
        min_index = np.argmin(df_this['Mood'])
        pymer_input = pymer_input.drop(df_this.index[min_index+1:])
    return pymer_input


# Analyzing change in boredom requires before-and-after group
batch = 'BoredomBeforeAndAfter'

print(f'=== Batch {batch}: Comparing LME models with and without deltaBoredom ===')
# Load pymer input table
in_file = f'{results_dir}/Mmi-{batch}_pymerInput-full.csv'
print(f'Opening {in_file}...')
pymer_input = pd.read_csv(in_file, index_col=0)


# Crop to exclude all mood ratings after miniumum rating (for floor effects)
pymer_input = CropToMinRating(pymer_input)

# Add deltaBoredom scores
in_file = f'{results_dir}/Mmi-{batch}_Probes.csv'
print(f'Opening {in_file}...')
df_boredom = pd.read_csv(in_file)
participants = np.unique(df_boredom.participant)
for participant_index,participant in enumerate(participants):
    # crop to this participant
    df_this = df_boredom.loc[df_boredom.participant==participant,:]
    # get change in boredom scores
    initial_boredom = np.sum(df_this.loc[df_this.iBlock==-1,'rating']) # after first block
    final_boredom = np.sum(df_this.loc[df_this.iBlock==0,'rating']) # after first block
    # add to pymer_input table
    pymer_input.loc[pymer_input.Subject==participant,'deltaBoredom'] = final_boredom - initial_boredom

# Fit models and run ANOVA to compare
lm_string_h0 = 'Mood ~ 1 + Time * (isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)'
lm_string_h1 = 'Mood ~ 1 + Time * (deltaBoredom + isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)'
anova_res, dfFit_h0, dfFit_h1 = CompareModels(pymer_input,lm_string_h0,lm_string_h1)

# Print results and fit
if anova_res.loc[1,'Pr(>Chisq)']<0.05:
    print('** (PRE-MIN ONLY) Change in state boredom DOES explain added variance in subject-level POTD slope.')
else:
    print('** (PRE-MIN ONLY) Change in state boredom does NOT explain added variance in subject-level POTD slope.')
print(dfFit_h1.loc[['deltaBoredom','Time:deltaBoredom']])


# %% Hyp 4.2.2: Floor effects in Effect of finalEmoDim on mood
print('=======================================')
print('')
print('=======================================')
print("""
4.2.2) We hypothesize that the final emotion dimension score will explain 
    variance in subject-level POTD slope. This is a one-sided hypothesis.
    We will test this with an ANOVA comparing the following two mixed effects 
    models (difference highlighted in bold):
    H0: Mood ~ 1 + Time * (isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)
    H1: Mood ~ 1 + Time * (finalEmoDim + isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)
""")

# Because 2.1 was not significant, use both before-and-after and after-only batches.
batch = 'AllMw'
# If 2.1 were not significant, we'd use after-only batch.
# batch = 'MwAfterOnly'

print(f'=== Batch {batch}: Comparing LME models with and without finalEmoDim ===')
in_file = f'{results_dir}/Mmi-{batch}_pymerInput-full.csv'
print(f'Opening {in_file}...')
pymer_input = pd.read_csv(in_file, index_col=0)

# Crop to exclude all mood ratings after miniumum rating (for floor effects)
pymer_input = CropToMinRating(pymer_input)

# Add finalEmoDim scores
in_file = f'{results_dir}/Mmi-{batch}_Probes.csv'
print(f'Opening {in_file}...')
df_mw = pd.read_csv(in_file)
participants = np.unique(df_mw.participant)
for participant_index,participant in enumerate(participants):
    # crop to this participant
    df_this = df_mw.loc[df_mw.participant==participant,:]
    # get final MW score
    X_this = np.atleast_2d(df_this.loc[df_this.iBlock==0,'rating'])-0.5 # iBlock==0: after first block. -0.5: Move center of scale to 0
    final_mw = pca.transform(X_this)[0,emotion_pc_index] 
    # add to pymer_input table
    pymer_input.loc[pymer_input.Subject==participant,'finalEmoDim'] = final_mw

# Fit models and run ANOVA to compare
lm_string_h0 = 'Mood ~ 1 + Time * (isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)'
lm_string_h1 = 'Mood ~ 1 + Time * (finalEmoDim + isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)'
anova_res, dfFit_h0, dfFit_h1 = CompareModels(pymer_input,lm_string_h0,lm_string_h1)

# Print results and fit
if anova_res.loc[1,'Pr(>Chisq)']<0.05:
    print('** (PRE-MIN ONLY) Final MW emotion DOES explain added variance in subject-level POTD slope.')
else:
    print('** (PRE-MIN ONLY) Final MW emotion does NOT explain added variance in subject-level POTD slope.')
print(dfFit_h1.loc[['finalEmoDim','Time:finalEmoDim']])
    

# %% Hyp 4.2.3: Floor effects in Effect of deltaEmoDim on mood
print('=======================================')
print('')
print('=======================================')
print("""
4.2.3) We hypothesize that the change in emotion dimension score will explain 
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

# Analyzing change in MW requires before-and-after group
batch = 'MwBeforeAndAfter'

print(f'=== Batch {batch}: Comparing LME models with and without deltaEmoDim ===')
in_file = f'{results_dir}/Mmi-{batch}_pymerInput-full.csv'
print(f'Opening {in_file}...')
pymer_input = pd.read_csv(in_file, index_col=0)

# Crop to exclude all mood ratings after miniumum rating (for floor effects)
pymer_input = CropToMinRating(pymer_input)

# Add deltaEmoDim scores
in_file = f'{results_dir}/Mmi-{batch}_Probes.csv'
print(f'Opening {in_file}...')
df_mw = pd.read_csv(in_file)
participants = np.unique(df_mw.participant)
for participant_index,participant in enumerate(participants):
    # crop to this participant
    df_this = df_mw.loc[df_mw.participant==participant,:]
    # get change in MW score
    X_this = np.atleast_2d(df_this.loc[df_this.iBlock==-1,'rating'])-0.5 # Before 1st block. Move center of scale to 0
    initial_mw = pca.transform(X_this)[0,emotion_pc_index] 
    X_this = np.atleast_2d(df_this.loc[df_this.iBlock==0,'rating'])-0.5 # After 1st block. Move center of scale to 0
    final_mw = pca.transform(X_this)[0,emotion_pc_index] 
    # add to pymer_input table
    pymer_input.loc[pymer_input.Subject==participant,'deltaEmoDim'] = final_mw - initial_mw

# Fit models and run ANOVA to compare
lm_string_h0 = 'Mood ~ 1 + Time * (isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)'
lm_string_h1 = 'Mood ~ 1 + Time * (deltaEmoDim + isMale + meanIRIOver20 + fracRiskScore + isAge40to100) + (1 + Time|Subject)'
anova_res, dfFit_h0, dfFit_h1 = CompareModels(pymer_input,lm_string_h0,lm_string_h1)

# Print results and fit
if anova_res.loc[1,'Pr(>Chisq)']<0.05:
    print('** (PRE-MIN ONLY) Change in MW emotion DOES explain added variance in subject-level POTD slope.')
else:
    print('** (PRE-MIN ONLY) Change in MW emotion does NOT explain added variance in subject-level POTD slope.')
print(dfFit_h1.loc[['deltaEmoDim','Time:deltaEmoDim']])
