#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RunCohortMeanLmers.py

Check extent to which cohort mean mood over time can be modeled by an LME.

- Created 9/14/22 by DJ
"""

# import packages
# import CompareTwoLmers as c2l
import pandas as pd
import numpy as np
import MoodDrift.Analysis.PlotMmiData as pmd
from matplotlib import pyplot as plt

# declare variables
results_dir = '../Data/OutFiles'
batch = 'AllOpeningRestAndRandom'

# load pymer input table
in_file = f'{results_dir}/Mmi-{batch}_pymerInput-full.csv'
print(f'Opening {in_file}...')
pymer_input = pd.read_csv(in_file, index_col=0)

# prepare to switch column names to what GetMeanRatings is expecting
pymer_columns = np.array(pymer_input.columns)
cohort_columns = pymer_columns.copy()
cohort_columns[cohort_columns=='Subject'] = 'participant'
cohort_columns[cohort_columns=='Mood'] = 'rating'
cohort_columns[cohort_columns=='Time'] = 'time'

# set up loop
cohorts = np.unique(pymer_input.Cohort) # get cohorts in this file
cohort_pymer_input = pd.DataFrame(columns=['Cohort','Mood','Time']) # initialize empty dataframe
for cohort in cohorts:
    # extract data from this cohort
    this_cohort = pymer_input.loc[pymer_input.Cohort==cohort].copy()
    # switch column names to what GetMeanRatings is expecting
    this_cohort.columns = cohort_columns
    this_cohort['iBlock'] = 0
    this_cohort['iTrial'] = this_cohort['iRating']
    # get mean time and rating for each rating in this cohort
    cohort_mean = pmd.GetMeanRatings(this_cohort,doInterpolation=False)
    # switch column names back to what pymer is expecting
    cohort_mean['Cohort'] = cohort
    cohort_mean = cohort_mean.loc[:,['Cohort','rating','time']]
    cohort_mean.columns = ['Cohort','Mood','Time']
    # append result to new cohort-level pymer input
    cohort_pymer_input = cohort_pymer_input.append(cohort_mean)

# %% Plot to illustrate the difference between what's being modeled
plt.figure(1)
plt.subplot(2,1,1)
participants = np.unique(pymer_input.Subject)
for participant in participants:
    these_ratings = pymer_input.loc[pymer_input.Subject==participant,:]
    plt.plot(these_ratings.Time,these_ratings.Mood,'b',alpha=0.02)

plt.xlabel('Time (min)')
plt.ylabel('Mood (0-1)')
plt.title('1 line per participant')

plt.subplot(2,1,2)
for cohort in cohorts:
    these_ratings = cohort_pymer_input.loc[cohort_pymer_input.Cohort==cohort,:]
    plt.plot(these_ratings.Time,these_ratings.Mood,'b',alpha=0.2)
plt.xlabel('Time (min)')
plt.ylabel('Mood (0-1)')
plt.title('1 line per cohort')
plt.tight_layout()
plt.show()


# %% Run LMEs
import CompareTwoLmers as c2l

# declare model strings
lm_string_h0 = 'Mood ~ 1 + (1 + Time|Cohort)'
lm_string_h1 = 'Mood ~ 1 + Time + (1 + Time|Cohort)'

lm_string_null = 'Mood ~ 1 + (1 + Time|Cohort)'


# Run ANOVA
anova_res, dfFit_h0, dfFit_h1, dfFixef_h0 = c2l.compare_lmers(cohort_pymer_input,lm_string_h0,lm_string_h1, lm_string_null)

# Print results
c2l.print_comparison_results(batch,cohort_pymer_input,lm_string_h0,lm_string_h1,anova_res,dfFit_h0,dfFit_h1)
