#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GetMmiIcc.py
Created on Fri Sep 18 12:22:26 2020

@author: jangrawdc

- Updated 10/29/20 by DJ - made axis labels have only first letter capitalized
- Updated 3/31/21 by DJ - adapted for shared code structure.
- Updated 4/15/21 by DJ - adjusted to new key filenames
"""


# %%
# conda install -c r rpy2
# conda install r-psych
import rpy2
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
pandas2ri.activate()
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
psych = importr('psych')
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

def PlotReliability(cohort1,cohort2,intOrSlope='Intercept',dataDir='../Data/OutFiles'):

    # load data
    dfKey = pd.read_excel('%s/%s_key.xlsx'%(dataDir,cohort2),engine='openpyxl')

    dfCoeff1 = pd.read_csv('%s/Mmi-%s_pymerCoeffs-full.csv'%(dataDir,cohort1))
    dfCoeff2 = pd.read_csv('%s/Mmi-%s_pymerCoeffs-full.csv'%(dataDir,cohort2))
    for i in range(dfCoeff1.shape[0]):
        subj1 = dfCoeff1.loc[i,'Subject']
        try:
            subj2 = dfKey.loc[dfKey.participant_day1==subj1,'participant_day2'].values[0]
            dfCoeff1.loc[i,'Subject2'] = subj2
            dfCoeff1.loc[i,'(Intercept)2'] = dfCoeff2.loc[dfCoeff2.Subject==subj2,'(Intercept)'].values[0]
            dfCoeff1.loc[i,'Time2'] = dfCoeff2.loc[dfCoeff2.Subject==subj2,'Time'].values[0]
        except:
            pass

    print('Dropping %d/%d missing lines and many irrelevant columns...'%(np.sum(pd.isna(dfCoeff1.Time2)),dfCoeff1.shape[0]))
    colsToKeep = ['(Intercept)','(Intercept)2','Time','Time2']
    dfCoeff = dfCoeff1.loc[pd.notna(dfCoeff1.Time2),colsToKeep].reset_index(drop=True)
    dfCoeff.columns = ['Intercept1','Intercept2','Slope1','Slope2']

    # Plot
    ax = plt.gca()
    plt.plot(dfCoeff['%s1'%intOrSlope],dfCoeff['%s2'%intOrSlope],'.')
    # Add 1:1 line
    minmax = [min(ax.get_xlim()[0],ax.get_ylim()[0]), \
              max(ax.get_xlim()[1],ax.get_ylim()[1])]
    plt.plot(minmax,minmax,'k:')
    # Annotate plot
    ax.set_ylim(minmax)
    ax.axis('square')
    plt.grid(True)
    # Add axis labels
    if cohort2=='Stability01-Rest_block2':
        ax.set_xlabel('%s (block 1)'%intOrSlope)
        ax.set_ylabel('%s (block 2)'%intOrSlope)
    elif (cohort2=='Stability02-Rest') or (cohort2=='Stability02-RandomVer2'):
        ax.set_xlabel('%s (day 1)'%intOrSlope)
        ax.set_ylabel('%s (day 2)'%intOrSlope)
    elif cohort2=='COVID02':
        ax.set_xlabel('%s (week 1)'%intOrSlope)
        ax.set_ylabel('%s (week 2)'%intOrSlope)
    elif cohort2=='COVID03':
        ax.set_xlabel('%s (week 1)'%intOrSlope)
        ax.set_ylabel('%s (week 3)'%intOrSlope)


def GetIcc(dfInput):

    #dfCoeffR = pandas2ri.py2ri(dfInput)
    #dfCoeffR = numpy2ri.py2ri(dfCoeff[['Intercept1','Intercept2']].values)

    icc = psych.ICC(dfInput)

    # with localconverter(ro.default_converter + pandas2ri.converter):
    #    pd_from_r_df = ro.conversion.rpy2py(icc[0])

    # colnames = np.zeros(len(icc[0][0]),dtype=object)
    # for i in range(len(icc[0][0])):
    #     colnames[i] = icc[0][0][i]
    # colnames = ['Single_raters_absolute','Single_random_raters','Single_fixed_raters',
    #             'Average_raters_absolute','Average_random_raters','Average_fixed_raters']
    # colnames = icc[0].index
    # rownames = ['ICC','F','df1','df2','p','CImin','CImax']

    # data = np.zeros((len(icc[0])-1,len(icc[0][1])))
    # for i in range(1,len(icc[0])):
    #     for j in range(len(icc[0][i])):
    #         data[i-1,j] = icc[0][i][j]
    dfICC = icc[0]
    icc21 = dfICC.loc[dfICC.type == 'ICC2','ICC']
    p21 = dfICC.loc[dfICC.type == 'ICC2','p']

    return icc21, p21




def GetMmiIcc(cohort1,cohort2,doPlot=False,dataDir='../Data/OutFiles',outFigDir='../Figures'):
    # load data
    dfKey = pd.read_excel('%s/%s_key.xlsx'%(dataDir,cohort2),engine='openpyxl')

    dfCoeff1 = pd.read_csv('%s/Mmi-%s_pymerCoeffs-full.csv'%(dataDir,cohort1))
    dfCoeff2 = pd.read_csv('%s/Mmi-%s_pymerCoeffs-full.csv'%(dataDir,cohort2))
    for i in range(dfCoeff1.shape[0]):
        subj1 = dfCoeff1.loc[i,'Subject']
        try:
            subj2 = dfKey.loc[dfKey.participant_day1==subj1,'participant_day2'].values[0]
            dfCoeff1.loc[i,'Subject2'] = subj2
            dfCoeff1.loc[i,'(Intercept)2'] = dfCoeff2.loc[dfCoeff2.Subject==subj2,'(Intercept)'].values[0]
            dfCoeff1.loc[i,'Time2'] = dfCoeff2.loc[dfCoeff2.Subject==subj2,'Time'].values[0]
        except:
            pass
#            print('subject %d (%d) not found in key.'%(i,subj1))
#            dfCoeff1.drop(index=i)

    print('Dropping %d/%d missing lines and many irrelevant columns...'%(np.sum(pd.isna(dfCoeff1.Time2)),dfCoeff1.shape[0]))
    colsToKeep = ['(Intercept)','(Intercept)2','Time','Time2']
    dfCoeff = dfCoeff1.loc[pd.notna(dfCoeff1.Time2),colsToKeep].reset_index(drop=True)
    dfCoeff.columns = ['Intercept1','Intercept2','Slope1','Slope2']



    # %% calculate ICC
    icc21_int, p21_int = GetIcc(dfCoeff[['Intercept1','Intercept2']])
    icc21_slope, p21_slope = GetIcc(dfCoeff[['Slope1','Slope2']])

    # Print results
    print('=== %s vs. %s ==='%(cohort1,cohort2))
#    print(icc[0])
#    print(dfICC)
    print('Intercept: ICC(2,1)=%.3g, p=%.3g'%(icc21_int,p21_int))
    print('Time: ICC(2,1)=%.3g, p=%.3g'%(icc21_slope,p21_slope))



    # %% Plot
    if doPlot == '2D':
        plt.figure(923,figsize=(8,5)); plt.clf()
        ax1 = plt.subplot(121);
        plt.plot(dfCoeff['Intercept1'],dfCoeff['Intercept2'],'.')
        minmax = [min(ax1.get_xlim()[0],ax1.get_ylim()[0]), \
                  max(ax1.get_xlim()[1],ax1.get_ylim()[1])]
        plt.plot(minmax,minmax,'k:')
        ax1.set_ylim(minmax)
        ax1.axis('square')
        plt.grid(True)
#        rs,p = stats.spearmanr(dfCoeff['Intercept1'],dfCoeff['Intercept2'],nan_policy='omit')
#        plt.title('LME Intercept Coefficient\n'+r'r_s=%.3g, p=%.3g'%(rs,p))
        plt.title('LME intercept coefficient\n'+r'ICC(2,1)=%.3g, p=%.3g'%(icc21_int,p21_int))

        ax2 = plt.subplot(122);
        plt.plot(dfCoeff['Slope1'],dfCoeff['Slope2'],'.')
        minmax = [min(ax2.get_xlim()[0],ax2.get_ylim()[0]), \
                  max(ax2.get_xlim()[1],ax2.get_ylim()[1])]
        plt.plot(minmax,minmax,'k:')
        ax2.set_ylim(minmax)
        ax2.axis('square')
        plt.grid(True)
#        rs,p = stats.spearmanr(dfCoeff['Slope1'],dfCoeff['Slope2'],nan_policy='omit')
#        plt.title('LME Slope Coefficient\n'+r'r_s=%.3g, p=%.3g'%(rs,p))
        plt.title('LME slope coefficient\n'+r'ICC(2,1)=%.3g, p=%.3g'%(icc21_slope,p21_slope))

        if cohort2=='Stability01-Rest_block2':
            ax1.set_xlabel('Block 1')
            ax1.set_ylabel('Block 2')
            ax2.set_xlabel('Block 1')
            ax2.set_ylabel('Block 2')
            plt.suptitle('Daily-Rest cohort (Rest block, 1 gambling block apart)')
        elif cohort2=='Stability02-Rest':
            ax1.set_xlabel('Day 1')
            ax1.set_ylabel('Day 2')
            ax2.set_xlabel('Day 1')
            ax2.set_ylabel('Day 2')
            plt.suptitle('Daily-Rest cohort (Rest block, 1 day apart)')
        elif cohort2=='Stability02-RandomVer2':
            ax1.set_xlabel('Day 1')
            ax1.set_ylabel('Day 2')
            ax2.set_xlabel('Day 1')
            ax2.set_ylabel('Day 2')
            plt.suptitle('Daily-Random cohort (random gambling block, 1 day apart)')
        elif cohort2=='COVID02':
            ax1.set_xlabel('Week 1')
            ax1.set_ylabel('Week 2')
            ax2.set_xlabel('Week 1')
            ax2.set_ylabel('Week 2')
            plt.suptitle('Weekly-Rest cohort (rest block, 1 week apart)')
        elif cohort2=='COVID03':
            ax1.set_xlabel('Week 1')
            ax1.set_ylabel('Week 3')
            ax2.set_xlabel('Week 1')
            ax2.set_ylabel('Week 3')
            plt.suptitle('Weekly-Rest cohort (rest block, 2 weeks apart)')

        plt.tight_layout(rect=(0,0,1.0,0.94))
        outFile = '%s/Mmi_%s-%s_reliability.png'%(outFigDir,cohort1,cohort2)
        print('Saving figure as %s...'%outFile)
        plt.savefig(outFile)
        outFile = '%s/Mmi_%s-%s_reliability.pdf'%(outFigDir,cohort1,cohort2)
        print('Saving figure as %s...'%outFile)
        plt.savefig(outFile)
        print('Done!')

    elif doPlot=='T1T2':
        plt.figure(923); plt.clf()
        ax1 = plt.subplot(121);
        plt.plot([0,1],dfCoeff.loc[:,['Intercept1','Intercept2']].T,'.-')
        plt.grid(True)
        plt.title('LME intercept coefficient\n'+r'ICC(2,1)=%.3g, p=%.3g'%(icc21_int,p21_int))
        plt.xticks([0,1],['T1','T2'])
        plt.ylabel('Intercept (mood, 0-1)')

        ax2 = plt.subplot(122);
        plt.plot([0,1],dfCoeff.loc[:,['Slope1','Slope2']].T,'.-')
        plt.grid(True)
        plt.title('LME slope coefficient\n'+r'ICC(2,1)=%.3g, p=%.3g'%(icc21_slope,p21_slope))
        plt.xticks([0,1],['T1','T2'])
        plt.ylabel('Slope (mood/minute)')

        if cohort2=='Stability02-Rest':
            ax1.set_xticklabels(['Day 1','Day 2'])
            ax2.set_xticklabels(['Day 1','Day 2'])
            plt.suptitle('Daily-Rest cohort (rest block, 1 day apart)')
        elif cohort2=='Stability02-RandomVer2':
            ax1.set_xticklabels(['Day 1','Day 2'])
            ax2.set_xticklabels(['Day 1','Day 2'])
            plt.suptitle('Daily-Random cohort (random gambling block, 1 day apart)')
        elif cohort2=='COVID02':
            ax1.set_xticklabels(['Week 1','Week 2'])
            ax2.set_xticklabels(['Week 1','Week 2'])
            plt.suptitle('Weekly-Rest cohort (Rest block, 1 week apart)')
        elif cohort2=='COVID03':
            ax1.set_xticklabels(['Week 1','Week 3'])
            ax2.set_xticklabels(['Week 1','Week 3'])
            plt.suptitle('Weekly-Rest cohort (rest block, 2 weeks apart)')


        plt.tight_layout(rect=(0,0,1.0,0.94))
        outFile = '%s/Mmi_%s-%s_T1T2.png'%(outFigDir,cohort1,cohort2)
        print('Saving figure as %s...'%outFile)
        plt.savefig(outFile)
        outFile = '%s/Mmi_%s-%s_T1T2.pdf'%(outFigDir,cohort1,cohort2)
        print('Saving figure as %s...'%outFile)
        plt.savefig(outFile)
        print('Done!')

    # %% Return
    return icc21_int,p21_int,icc21_slope,p21_slope
