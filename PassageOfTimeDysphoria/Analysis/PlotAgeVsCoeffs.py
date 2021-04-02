#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PlotAgeVsCoeffs.py
Plot participant age against their LME coefficients
Created on Fri Sep 18 10:11:34 2020

@author: jangrawdc

- Updated 10/29/20 by DJ - made axis labels have only first letter capitalized
- Updated 3/31/21 by DJ - adapted for shared code structure.
"""

# %% Set up
import pandas as pd
from matplotlib import pyplot as plt

# Declare directories & constants
dataDir = '../Data/OutFiles'
outFigDir = '../Figures'
cohort = 'AllOpeningRestAndRandom'

# %% Load data
dfCoeffs = pd.read_csv('%s/Mmi-%s_pymerCoeffs-full-noAge.csv'%(dataDir,cohort))
dfAll = pd.read_csv('%s/Mmi-%s_pymerInputWithAges.csv'%(dataDir,cohort)) # or get from Run... script
nSubj = dfCoeffs.shape[0]
for i in range(nSubj):
    dfCoeffs.loc[i,'age'] = dfAll.loc[dfAll.Subject==dfCoeffs.Subject[i],'age'].values[0]
    dfCoeffs.loc[i,'Cohort'] = dfAll.loc[dfAll.Subject==dfCoeffs.Subject[i],'Cohort'].values[0]
    dfCoeffs.loc[i,'fracRiskScore'] = dfAll.loc[dfAll.Subject==dfCoeffs.Subject[i],'fracRiskScore'].values[0]

# %% Plot age vs. slope, colored by Cohort

plt.figure(342); plt.clf();
isNimh = dfCoeffs.Cohort=='RecoveryNimh-run1'
plt.subplot(2,1,1)
plt.plot(dfCoeffs.loc[isNimh,'age'],dfCoeffs.loc[isNimh,'Time']*100.0,'r.',label='NIMH in-person')
plt.plot(dfCoeffs.loc[~isNimh,'age'],dfCoeffs.loc[~isNimh,'Time']*100.0,'b.',label='mTurk')
plt.xlabel('Age (years)')
plt.ylabel('Slope (% mood/min)')
plt.title('Age vs. LME slope parameter in online participants')
plt.legend()
#plt.colorbar()

plt.subplot(2,1,2)
plt.scatter(dfCoeffs['age'],dfCoeffs['Time']*100.0,c=dfCoeffs['fracRiskScore'],marker='.',label='participant')
plt.colorbar().set_label('Fraction of depression risk score\n(MFQ/12 or CESD/16)')
plt.xlabel('Age (years)')
plt.ylabel('Slope (% mood/min)')
plt.tight_layout()

plt.savefig('%s/Mmi-%s_ageVsSlope.png'%(outFigDir,cohort))
