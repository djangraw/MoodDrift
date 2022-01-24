import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.weightstats import ttost_ind
from matplotlib import pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.anova as anova
from pymer4.models import Lmer
from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression


# need distribution of CES-D scores
svdat = pd.read_csv('../Data/OutFiles/Mmi-AdultOpeningRest_Survey.csv', index_col=0)
trdat = pd.read_csv('../Data/OutFiles/Mmi-AdultOpeningRest_Trial.csv', index_col=0)


# Create indicator for block type
trdat.loc[(trdat.trialType=='closed') & (trdat.targetHappiness == 'random'), 'targetHappiness'] = -1
trdat['targetHappiness'] = trdat.targetHappiness.astype(float)

trdat['trial_type'] = trdat.trialType
trdat.loc[(trdat.trialType=='closed') & (trdat.targetHappiness == 1), 'trial_type'] = 'positive'
trdat.loc[(trdat.trialType=='closed') & (trdat.targetHappiness == 0), 'trial_type'] = 'negative'
trdat.loc[(trdat.trialType=='closed') & (trdat.targetHappiness == -1), 'trial_type'] = 'random'

# get minimum ratings during rest and negative blocks
neg_batches = (trdat
               .groupby(['batchName', 'iBlock', 'trial_type'])[['participant']]
               .count()
               .reset_index()
               .query("trial_type == 'negative'")
               .batchName.values)
rdat = (trdat
        .loc[trdat.batchName.isin(neg_batches) & trdat.trial_type.isin(['rest'])]
        .copy()
        .groupby(['batchName', 'participant', 'iBlock',])[['rating']]
        .min()
        .reset_index())
ndat = (trdat
        .loc[trdat.batchName.isin(neg_batches) & trdat.trial_type.isin(['negative'])]
        .copy()
        .groupby(['batchName', 'participant', 'iBlock',])[['rating']]
        .min()
        .reset_index())
rndat = rdat.merge(ndat, how='outer', on=['batchName', 'participant'], suffixes=['_rst', '_neg'])

print("batches with rest and negative blocks:")
print(rndat.batchName.unique())

# merge in fractional risk score
svdat['frs'] = svdat.CESD/16
assert svdat.frs.isnull().sum() == 0
rndat = rndat.merge(svdat.loc[:, ['participant', 'batchName', 'frs']], how='left', on=['batchName', 'participant'])

rndat['relative_floor'] = (rndat.rating_rst <= rndat.rating_neg).astype(int)
rndat['actual_floor'] = (rndat.rating_rst == 0).astype(int)
rndat['mindif'] = rndat.rating_rst - rndat.rating_neg
rndat['keep_floor'] = (~(rndat.relative_floor.astype(bool) | rndat.relative_floor.astype(bool) )).astype(int)

# indicators to drop anyone who's hit a relative or absolute floor
floor_kept = rndat.loc[rndat.keep_floor == 1, ['batchName', 'participant']]
floor_kept = floor_kept.rename(columns={'batchName':'Cohort', 'participant':'Subject'})
floor_dropped = rndat.loc[rndat.keep_floor == 0, ['batchName', 'participant']]

# Fit lmers
pymer_input = pd.read_csv('../Data/OutFiles_rb/Mmi-AllOpeningRestAndRandom_pymerInput-full.csv', index_col=0)

# basic result
print("basic result")
lmString = 'Mood ~ 1 + Time * (isMale + meanIRIOver20 + totalWinnings + meanRPE + fracRiskScore + isAge0to16 + ' \
           'isAge16to18 + isAge40to100) + (Time | Subject)'
modela = Lmer(lmString, data=pymer_input)
_ = modela.fit()
dfFit = modela.coefs
print(dfFit)

# just drop actual floor and rerun
print()
print("Just drop actual floor and rerun")
act_floor_dat = pymer_input.groupby(['Cohort', 'Subject'])[['Mood', 'fracRiskScore']].min().reset_index()
act_floor_dat['actual_floor'] = (act_floor_dat.Mood == 0).astype(int)
no_floor_df = act_floor_dat.loc[act_floor_dat.actual_floor==0, ['Cohort', 'Subject']]
pymer_noactfloor = no_floor_df.merge(pymer_neg_subs, how='inner', on=['Cohort', 'Subject'])
lmString = 'Mood ~ 1 + Time * (isMale + meanIRIOver20 + totalWinnings + meanRPE + fracRiskScore + isAge0to16 + ' \
           'isAge16to18 + isAge40to100) + (Time | Subject)'
model = Lmer(lmString, data=pymer_noactfloor)
_ = model.fit()
dfFit = model.coefs
print(dfFit)
print("Interaction is still there.")
print()

# Do we see the Time:fracRiskScore interaction in the cohorts where we can calculate a relative floor?
print("Do we see the Time:fracRiskScore interaction in the cohorts where we can calculate a relative floor?")
pymer_floor_cohorts = pymer_input.loc[pymer_input.Cohort.isin(floor_kept.Cohort.unique())]
lmString = 'Mood ~ 1 + Time * (isMale + meanIRIOver20 + totalWinnings + meanRPE + fracRiskScore + isAge0to16 + ' \
           'isAge16to18 + isAge40to100) + (Time | Subject)'
modela = Lmer(lmString, data=pymer_floor_cohorts)
_ = modela.fit()
dfFit = modela.coefs
print(dfFit)
print("We don't, so we can't use relative floor.")

# calc a GLM for each person
pymer_input
lmString = 'Mood ~ 1 + Time'
glmres = []
for ss, df in pymer_input.groupby('Subject'):
    try:
        res = smf.glm(lmString, df).fit()
        row = res.params
        row['Subject'] = ss
        row['fracRiskScore'] = df.fracRiskScore.unique()[0]
        row['frs_bucket'] = df.frs_bucket.unique()[0]

        glmres.append(row)
    except ValueError:
        continue
glmres = pd.DataFrame(glmres)

# binarize time
glmres['posTime'] = (glmres.Time > 0).astype(int)
print("Proportion of people with a positive mood slope over time.")
print(glmres.groupby('frs_bucket').posTime.mean())

g = sns.lmplot(x='fracRiskScore', y='Time', data=glmres)
g.fig.savefig('../Figures/fracRiskScore_vs_time.png')

print("Pearson r between time slope and frac risk score.")
print(stats.pearsonr(glmres.Time, glmres.fracRiskScore))
print("Pearson r between time slope and frac risk score if you just look at positive slopes.")
print(stats.pearsonr(glmres.query("Time > 0").Time, glmres.query("Time > 0").fracRiskScore))
print()
pt_frs_obs = glmres.groupby(['frs_bucket', 'posTime']).Intercept.count().reset_index().pivot(index='frs_bucket',
                                                                                             columns='posTime')
print("Positive time slope vs fractional risk score observed contengency")
print(pt_frs_obs)
print()
chi2, p, dof, expected = stats.chi2_contingency(pt_frs_obs)
print("Positive time slope vs fractional risk score expected contengency")
print(expected)
print(f'chi2_{dof} = {chi2}, p = {p}')

pos_subs = glmres.loc[glmres.posTime == 1, 'Subject'].astype(int).astype(str).values

# do we still see the Time:fracRiskScore interaction in the subset of people with a positive mood slope
print()
print("Do we still see the Time:fracRiskScore interaction in the subset of people with a positive mood slope?")
pymer_pos_subs = pymer_input.loc[pymer_input.Subject.isin(pos_subs)]
lmString = 'Mood ~ 1 + Time * (isMale + meanIRIOver20 + totalWinnings + meanRPE + fracRiskScore + isAge0to16 + ' \
           'isAge16to18 + isAge40to100) + (Time | Subject)'
modela = Lmer(lmString, data=pymer_pos_subs)
_ = modela.fit()
dfFit = modela.coefs
print(dfFit)

# It could be a ceiling effect on people without depression
# Let's drop anyone who had a mood rating of 0.7 or higher, which should also account for subjective ceilings
print("It could be a ceiling effect on people without depression.")
print("Let's drop anyone who had a mood rating of 0.7 or higher, which should also account for subjective ceilings.")

act_ceil_dat = pymer_input.groupby(['Cohort', 'Subject'])[['Mood', 'fracRiskScore']].max().reset_index()
act_ceil_dat['actual_ceil'] = (act_ceil_dat.Mood > 0.7).astype(int)
no_ceil_df = act_ceil_dat.loc[act_ceil_dat.actual_ceil==0, ['Cohort', 'Subject']]
pymer_noactceil = no_ceil_df.merge(pymer_pos_subs, how='inner', on=['Cohort', 'Subject'])
lmString = 'Mood ~ 1 + Time * (isMale + meanIRIOver20 + totalWinnings + meanRPE + fracRiskScore + isAge0to16 + isAge16to18 + isAge40to100) + (Time | Subject)'
modela = Lmer(lmString, data=pymer_noactceil)
_ = modela.fit()
dfFit = modela.coefs
print(dfFit)
print("Looks like the interaction remains even in this case.")
print()

# look at last mood - first mood delta in addition to slopes
print("Proportion of people with a positive mood delta.")
print(first_and_last.groupby('frs_bucket').Mood_sign.mean())

g = sns.lmplot(x='fracRiskScore', y='Mood_delta', data=first_and_last)
g.fig.savefig('../Figures/fracRiskScore_vs_mooddelta.png')

print("Pearson r between mood delta and frac risk score.")
print(stats.pearsonr(first_and_last.Mood_delta, first_and_last.fracRiskScore))
print("Pearson r between mood delta and frac risk score if you just look at positive slopes.")
print(stats.pearsonr(first_and_last.query("Mood_delta > 0").Mood_delta, first_and_last.query("Mood_delta > 0").fracRiskScore))
print()
pt_frs_obs = first_and_last.groupby(['frs_bucket', 'Mood_sign']).Subject.count().reset_index().pivot(index='frs_bucket',columns='Mood_sign')
print("Positive mood delta vs fractional risk score observed contengency")
print(pt_frs_obs)
print()
chi2, p, dof, expected = stats.chi2_contingency(pt_frs_obs)
print("Positive time slope vs fractional risk score expected contengency")
print(expected)
print(f'chi2_{dof} = {chi2}, p = {p}')