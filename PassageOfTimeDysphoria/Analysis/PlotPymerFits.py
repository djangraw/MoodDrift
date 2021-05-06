#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 08:12:57 2020

@author: jangrawdc
"""

# Import common packages
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
import seaborn as sns


def AddFitLine(x,y):
    # Add fit line
    r,p = stats.spearmanr(x,y,nan_policy='omit')
    isOk = ~np.isnan(x) & ~np.isnan(y)
    m, b = np.polyfit(x[isOk], y[isOk], 1)
    plt.plot(x, m*x + b,label='linear fit (spearman $r=%.2f, p=%.2g$)'%(r,p))
    plt.legend()

def PlotPymerVsSurveys(dfCoeffs,dfSurvey,coeffToPlot='Time',colsToPlot=['gender','age','status','CESD','SHAPS','COVID']):

    # Combine coeffs & survey results
    dfCoeffs = dfCoeffs.rename(columns={"Subject": "participant"})
    #dfAll = dfAll.rename(columns={"Subject": "participant"})
    dfSurvey = dfSurvey.set_index('participant')
    dfCoeffs = dfCoeffs.set_index('participant')
    dfSurvey = pd.concat([dfSurvey,dfCoeffs],axis=1)

    # Make scatterplots
    nCols = max(3,np.ceil(np.sqrt(len(colsToPlot)))) # at least 3 columns
    nRows = np.ceil(1.0*len(colsToPlot)/nCols)

    # Plot
    for iPlot,col in enumerate(colsToPlot):
        # declare subplot
        plt.subplot(nRows,nCols,iPlot+1)
        # do histogram
        if col=='gender':
            xHist = np.linspace(np.nanmin(dfSurvey[coeffToPlot]),np.nanmax(dfSurvey[coeffToPlot]),11)
            x1 = dfSurvey.loc[(dfSurvey[col]=='Male') & pd.notna(dfSurvey[coeffToPlot]),coeffToPlot]
            x2 = dfSurvey.loc[(dfSurvey[col]=='Female') & pd.notna(dfSurvey[coeffToPlot]),coeffToPlot]
            plt.hist(x1,xHist,alpha=0.5,label='Male (n=%d)'%(x1.size));
            plt.hist(x2,xHist,alpha=0.5,label='Female (n=%d)'%(x2.size));
            z,p = stats.ranksums(x1,x2)
            # annotate
            plt.xlabel('%s coefficient\nranksum p=%.2g'%(coeffToPlot,p))
            plt.ylabel('# subjects')
            plt.legend()
        elif col=='diagnosis':
            xHist = np.linspace(np.nanmin(dfSurvey[coeffToPlot]),np.nanmax(dfSurvey[coeffToPlot]),11)
            x1 = dfSurvey.loc[(dfSurvey[col]=='HV') & pd.notna(dfSurvey[coeffToPlot]),coeffToPlot]
            x2 = dfSurvey.loc[(dfSurvey[col]=='MDD') & pd.notna(dfSurvey[coeffToPlot]),coeffToPlot]
            plt.hist(x1,xHist,alpha=0.5,label='HV (n=%d)'%(x1.size));
            plt.hist(x2,xHist,alpha=0.5,label='MDD (n=%d)'%(x2.size));
            z,p = stats.ranksums(x1,x2)
            # annotate
            plt.xlabel('%s coefficient\nranksum p=%.2g'%(coeffToPlot,p))
            plt.ylabel('# subjects')
            plt.legend()
        else:
            # make scatter plot
            plt.plot(dfSurvey[col],dfSurvey[coeffToPlot],'.',label='participant')
            # Add fit line
            AddFitLine(dfSurvey[col],dfSurvey[coeffToPlot])
            # annotate
            plt.xlabel(col)
            plt.ylabel('%s coefficient'%coeffToPlot)
    # Annotate figure
    plt.tight_layout(rect=[0,0,1,0.93]);
    # plt.suptitle('MMI batch %s (n=%d)'%(batchName,dfSurvey.shape[0]))


def PlotPymerHistos(dfCoeffs):
    '''
    Plot histograms of the intercept and slope coefficients.
    '''

    # Plot intercept histo
    plt.subplot(1,3,1);
    plt.hist(dfCoeffs['(Intercept)'])
    plt.xlabel('Intercept (mood at block start)')
    plt.ylabel('# subjects')
    plt.title('LME Intercept Parameter')

    plt.subplot(1,3,2);
    plt.hist(dfCoeffs['Time'])
    plt.xlabel('Slope '+ r'$(\Delta mood/min)$')
    plt.ylabel('# subjects')
    plt.title('LME Slope Parameter')

    plt.subplot(1,3,3);
    plt.plot(dfCoeffs['(Intercept)'],dfCoeffs['Time'],'.')
    # Add fit line
    AddFitLine(dfCoeffs['(Intercept)'],dfCoeffs['Time'])
    # annotate plot
    plt.xlabel('Intercept (mood at block start)')
    plt.ylabel('Slope '+ r'$(\Delta mood/min)$')
    plt.title('LME Parameters')


    # Annotate figure
    plt.tight_layout(rect=[0,0,1,0.93]);

def PlotPymerHistosJoint(dfCoeffs):
    '''
    Plot histograms and of the intercept and slope coefficients.
    '''

    # Plot intercept histo
#    with sns.axes_style("whitegrid"):
    g = sns.jointplot('(Intercept)','Time',
                      data=dfCoeffs[['(Intercept)','Time']],
                      kind="reg",space=0)
    # add axis labels
    g.ax_joint.set_xlabel("Initial mood parameter")
    g.ax_joint.set_ylabel("Mood slope parameter\n(mood/min)")
    # Add lines
    g.ax_joint.axvline(x=0.5,c='k',ls=':',zorder=-1,label='neutral')
    g.ax_joint.axhline(y=0,c='k',ls=':',zorder=-1,label='no change')
    g.ax_marg_x.axvline(x=0.5,c='k',ls=':',zorder=-1,label='neutral')
    g.ax_marg_y.axhline(y=0,c='k',ls=':',zorder=-1,label='no change')
    # add legend
    r,p = stats.spearmanr(dfCoeffs['(Intercept)'],dfCoeffs['Time'])
    phantom, = g.ax_joint.plot([], [], linestyle="", alpha=0)
    g.ax_joint.legend([phantom],[f'r={r:.3g}, p={p:.3g}'])

    # Annotate figure
    plt.tight_layout(rect=[0,0,1,0.93]);

    # return seaborn object
    return g
