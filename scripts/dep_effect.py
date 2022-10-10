import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.weightstats import ttost_ind
from matplotlib import pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
import statsmodels.stats.anova as anova
from pymer4.models import Lmer
from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from matplotlib.patches import Rectangle


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
pymer_input = pd.read_csv('../Data/OutFiles/Mmi-AllOpeningRestAndRandom_pymerInput-full.csv', index_col=0)

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
pymer_noactfloor = no_floor_df.merge(pymer_input, how='inner', on=['Cohort', 'Subject'])
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


pymer_input['Depression Risk'] = pymer_input.fracRiskScore >= 1
pymer_input.groupby('Depression Risk').Subject.nunique()


# calc a GLM for each person
lmString = 'Mood ~ 1 + Time'
glmres = []
for ss, df in pymer_input.groupby('Subject'):
    try:
        res = smf.glm(lmString, df).fit()
        row = res.params
        row['Subject'] = ss
        row['fracRiskScore'] = df.fracRiskScore.unique()[0]
        pvals = res.pvalues
        pvals.index = pvals.index + '_pval'
        row = pd.concat([row, pvals])
    except ValueError:
        assert df.Mood.nunique() == 1
        row = {}
        row['Subject'] = ss
        row['Intercept'] = df.Mood.unique()[0]
        row['Time'] = 0
        row['fracRiskScore'] = df.fracRiskScore.unique()[0]
        row['Intercept_pval'] = np.nan
        row['Time_pval'] = 1
    glmres.append(pd.Series(row))
glmres = pd.DataFrame(glmres)

alpha = 0.05
sig, _, _, _ = multipletests(glmres.Time_pval, alpha, method='fdr_bh')
glmres['sig'] = sig
glmres['Time Sign'] = 0
glmres.loc[(glmres.Time < 0) & (glmres.sig), 'Time Sign'] = -1
glmres.loc[(glmres.Time > 0) & (glmres.sig), 'Time Sign'] = 1
glmres['Depression Risk'] = glmres.fracRiskScore >= 1


assert glmres.Subject.nunique() == pymer_input.Subject.nunique()


pt_frs_obs = glmres.groupby(['Depression Risk', 'Time Sign']).Intercept.count().reset_index().pivot(index='Depression Risk', columns='Time Sign')
pt_frs_obs.columns = pt_frs_obs.columns.get_level_values(1)

# binarize time
print("Proportion of people with positive, zero, and negative mood slopes over time.")
print(pt_frs_obs)

print("Pearson r between time slope and frac risk score.")
print(stats.pearsonr(glmres.Time, glmres.fracRiskScore))
r, rp = stats.pearsonr(glmres.Time, glmres.fracRiskScore)
print("Pearson r between time slope and frac risk score if you just look at positive slopes.")
print(stats.pearsonr(glmres.query("Time > 0").Time, glmres.query("Time > 0").fracRiskScore))
print()


chi2, p, dof, expected = stats.chi2_contingency(pt_frs_obs)
pt_frs_exp = pt_frs_obs.copy()
pt_frs_exp.loc[False, :] = expected[0]
pt_frs_exp.loc[True, :] = expected[1]
print("Positive time slope vs fractional risk score expected contengency")
print(expected)
print(f'chi2_{dof} = {chi2}, p = {p}')


print("positive vs non-positive")
pt_frs_obs_pvnp = pt_frs_obs.copy()
pt_frs_obs_pvnp.loc[:, 0] = pt_frs_obs_pvnp.loc[:, -1] + pt_frs_obs_pvnp.loc[:, 0]
pt_frs_obs_pvnp = pt_frs_obs_pvnp.loc[:,[ 0,1]]
print("Proportion of people with positive and non-positive mood slopes over time.")
print(pt_frs_obs_pvnp)

chi2_pvnp, p_pvnp, dof_pvnp, expected_pvnp = stats.chi2_contingency(pt_frs_obs_pvnp)
pt_frs_exp_pvnp = pt_frs_obs_pvnp.copy()
pt_frs_exp_pvnp.loc[False, :] = expected_pvnp[0]
pt_frs_exp_pvnp.loc[True, :] = expected_pvnp[1]
print("Positive time slope vs fractional risk score expected contengency")
print(expected_pvnp)
print(f'chi2_{dof_pvnp} = {chi2_pvnp}, p = {p_pvnp}')

interped = []
for ss, df in pymer_input.groupby("Subject"):
    df['on_interval'] = False
    add_time = []
    for tt in np.arange(1/6, df.Time.max(), 1/6):
        row = df.iloc[0].copy()
        if tt not in df.Time.values:
            row['Time'] = tt
            row['Mood'] = np.nan
            row['on_interval'] = True
            add_time.append(row)
        else:
            df.loc[df.Time == tt, 'on_interval'] = True
    add_time = pd.DataFrame(add_time)
    timedf = pd.concat([df, add_time]).sort_values('Time')
    timedf['Timestamp'] = timedf.Time.apply(lambda x: pd.Timestamp(x, unit='m'))
    timedf = timedf.set_index('Timestamp')
    timedf['interp_mood'] = timedf.Mood.interpolate('time')
    timedf = timedf.reset_index()
    interped.append(timedf)
interped = pd.concat(interped).reset_index(drop=True)


time_to_plot = 6
subs_for_agg = interped.query('on_interval').groupby("Subject").Time.max().reset_index().query('Time >= @time_to_plot').Subject

time_agg = interped.query('on_interval & Time <= @time_to_plot & Subject.isin(@subs_for_agg)', engine='python').groupby(['Depression Risk', 'Time'])[['interp_mood']].agg(['mean', 'count', 'sem'])
time_agg.columns = time_agg.columns.get_level_values(1)
time_agg = time_agg.reset_index()

time_agg['leb'] = time_agg['mean'] - time_agg['sem']
time_agg['ueb'] = time_agg['mean'] + time_agg['sem']

to_plot= glmres.copy()
to_plot['Sign of Mood Drift'] = to_plot['Time Sign'].replace({-1:'Neg.', 0:'Non-sig.', 1:'Pos.'})


fig,axes = plt.subplots(1, 3, figsize=(7.5,5))
# Panel A
ax = axes[0]
nr_dat = time_agg.loc[~time_agg['Depression Risk']]
x = nr_dat.Time*60
y = nr_dat['mean']
ax.plot(x,y, label=f'Not at risk\n(n = {time_agg.loc[~time_agg["Depression Risk"], "count"].values[0]})')
y1 = nr_dat.leb
y2 = nr_dat.ueb
ax.fill_between(x, y1, y2, alpha = 0.4)

ar_dat = time_agg.loc[time_agg['Depression Risk']]
x = ar_dat.Time*60
y = ar_dat['mean']
ax.plot(x,y, label=f'At risk of depression\n(n = {time_agg.loc[time_agg["Depression Risk"], "count"].values[0]})')
y1 = ar_dat.leb
y2 = ar_dat.ueb
ax.fill_between(x, y1, y2, alpha = 0.4)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Mood rating (0-1)')
ax.legend(loc='upper center', bbox_to_anchor=(0.5,-0.25))
xlims = ax.get_xlim()
ax.plot(xlims,[0.5,0.5], color='black', linestyle='dashed')

# Panel B
ax = axes[1]
line_color = sns.color_palette()[4]
ns_color = sns.color_palette('pastel')[0]
sn_color = sns.color_palette('pastel')[3]
sp_color = sns.color_palette('pastel')[2]

glmres['Time'] = glmres['Time']*100 # convert to % mood/min
ax = sns.regplot(x='fracRiskScore', y='Time', data=glmres, scatter=False, line_kws={'color':line_color,
                                                                                  'label': 'Trend'}, ax=ax)
xns = glmres.loc[~glmres.sig,'fracRiskScore']
yns = glmres.loc[~glmres.sig,'Time']
ax.plot(xns, yns, '.', color=ns_color, zorder=-100, label=f'Non-significant\n(n = {len(xns)})')#, markerfacecolor="none")
xsn = glmres.loc[glmres['Time Sign'] == -1, 'fracRiskScore']
ysn = glmres.loc[glmres['Time Sign'] == -1, 'Time']
ax.plot(xsn, ysn, '.', color=sn_color, zorder=-100, label=f'Negative\n(n = {len(xsn)})')#, markerfacecolor="none")
xsp = glmres.loc[glmres['Time Sign'] == 1, 'fracRiskScore']
ysp = glmres.loc[glmres['Time Sign'] == 1, 'Time']
ax.plot(xsp, ysp, '.', color=sp_color, zorder=-100, label=f'Positive\n(n = {len(xsp)})')#, markerfacecolor="none")

xlims = ax.get_xlim()
ylims = ax.get_ylim()
r2_placeholder = Rectangle((0, 0), 1, 1, fc="w", fill=False,
                             edgecolor='none', linewidth=0)

# ax.plot(xlims, [0,0], 'black')

#ax.plot([1.0, 1.0], ylims, 'black')
ax.set_xticks((1, 2))
ax.set_xlim(xlims)
ax.set_ylim(ylims)
ax.set_ylabel('Mood drift (%mood/min)')
ax.set_xlabel("Depression risk")
ax.get_figure().set_facecolor('white')
handles, labels = ax.get_legend_handles_labels()
handles = [handles[2], handles[0], handles[1]]
labels = [labels[2], labels[0], labels[1]]

handles.append(ax.get_lines()[0])
#handles.append(r2_placeholder)
labels.append('Trend')
#labels.append(f'$r^2$ = {r**2:0.2f}\n$p$ = {rp:0.2g}')
ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5,-0.25))


# panel c
ax =axes[2]
normed = to_plot.groupby('Depression Risk')['Sign of Mood Drift'].value_counts(normalize=True).mul(100).rename('Percent').reset_index()
ax = sns.barplot(y='Percent', x='Sign of Mood Drift', hue='Depression Risk',
                data=normed,  order=['Pos.', 'Non-sig.', 'Neg.'],
                ax=ax)
pct_exp = pt_frs_exp.copy()
pct_exp.loc[False] = pt_frs_exp.loc[False] / pt_frs_exp.sum(1)[False] * 100
pct_exp.loc[True] = pt_frs_exp.loc[True] / pt_frs_exp.sum(1)[True] * 100
expectations = pct_exp.loc[:, [1,0,-1]].values.flatten()
first = True
for pp, exp in zip(ax.patches, expectations):
    x = [exp]*2
    y = [pp.get_x(), pp.get_x() + pp.get_width()]
    if first:
        first = False
        label = 'Expected'
    else:
        label = None
    ax.plot(y, x, color='black', linestyle='dashed', label=label)

#leg = ax.get_legend()
#leg.set_visible(False)

chi2_placeholder = Rectangle((0, 0), 1, 1, fc="w", fill=False,
                             edgecolor='none', linewidth=0)

handels, labels = ax.get_legend_handles_labels()

labels = [f'Not at risk\n(n = {(~glmres["Depression Risk"]).sum()})',
          f'At risk of depression\n(n = {(glmres["Depression Risk"]).sum()})',
          'Expected',
          f'$\chi^2$ = {chi2:0.2f}\n$p$ = {p:0.2g}']
handels = [handels[1], handels[2], handels[0], chi2_placeholder]
ax.legend(handels, labels, loc='upper center', bbox_to_anchor=(0.5,-0.25), )
ax.set_ylabel('Participants')

fig.tight_layout(pad=1, h_pad=0.1)
fig.set_facecolor('white')
fig.text(0, 0.95, 'A', weight='bold')
fig.text(0.33, 0.95, 'B', weight='bold')
fig.text(0.66, 0.95, 'C', weight='bold')

outFile = '../Figures/dep_effect'
print(f'Saving figure as {outFile}...')
fig.savefig(f'{outFile}.png', dpi=200,bbox_inches="tight")
fig.savefig(f'{outFile}.pdf')



#frs bucket vs delta mood instead of mood slope
pymer_input['Depression Risk'] = glmres.fracRiskScore >= 1
first_rate = pymer_input.sort_values(['Cohort', 'Subject', 'Time']).groupby(['Cohort', 'Subject'])[['Time', 'Mood', 'fracRiskScore', 'Depression Risk']].first().reset_index()
last_rate = pymer_input.sort_values(['Cohort', 'Subject', 'Time']).groupby(['Cohort', 'Subject'])[['Time', 'Mood']].last().reset_index()
first_and_last = first_rate.merge(last_rate, how='inner', on=['Cohort', 'Subject'], suffixes=['_first', '_last'])
first_and_last['Mood_delta'] = first_and_last.Mood_last - first_and_last.Mood_first
first_and_last['Mood_sign'] = np.sign(first_and_last.Mood_delta)

# look at last mood - first mood delta in addition to slopes
print("Proportion of people with a positive mood delta.")
print(first_and_last.groupby('Depression Risk').Mood_sign.mean())

g = sns.lmplot(x='fracRiskScore', y='Mood_delta', data=first_and_last)
g.fig.savefig('../Figures/fracRiskScore_vs_mooddelta.png')

print("Pearson r between mood delta and frac risk score.")
print(stats.pearsonr(first_and_last.Mood_delta, first_and_last.fracRiskScore))
print("Pearson r between mood delta and frac risk score if you just look at positive slopes.")
print(stats.pearsonr(first_and_last.query("Mood_delta > 0").Mood_delta, first_and_last.query("Mood_delta > 0").fracRiskScore))
print()
pt_frs_obs = first_and_last.groupby(['Depression Risk', 'Mood_sign']).Subject.count().reset_index().pivot(index='Depression Risk',columns='Mood_sign')
print("Positive mood delta vs fractional risk score observed contengency")
print(pt_frs_obs)
print()
chi2, p, dof, expected = stats.chi2_contingency(pt_frs_obs)
print("Positive time slope vs fractional risk score expected contengency")
print(expected)
print(f'chi2_{dof} = {chi2}, p = {p}')
