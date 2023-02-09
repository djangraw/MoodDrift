#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CalculatePytorchModelError.py
Created on Mon Oct  5 15:10:35 2020

@author: jangrawdc

Updated 10/29/20 by DJ - made axis labels have only first letter capitalized
Updated 3/31/21 by DJ - adapted for shared code structure.
Updated 4/23/21 by DJ - accept directories as inputs
Updated 9/30/22 by DJ - save svg figure, print median/IQRs
"""

# %%
# Set up
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
import seaborn as sns
from sklearn.kernel_ridge import KernelRidge

def CalculatePytorchModelError(IS_EXPLORE=False, IS_LATE=True, dataDir = '../Data/OutFiles', pytorchDir = '../Data/GbePytorchResults', outFigDir = '../Figures'):
    if IS_EXPLORE:
        suffix = '_GbeExplore'
    else:
        suffix = '_GbeConfirm'

    if IS_LATE:
        suffix = '%s-late'%suffix


    # Load results
    inFile = '%s/PyTorchPredictions%s.npy'%(pytorchDir,suffix)
    print('Loading pyTorch best fits from %s...'%inFile)
    fits = np.load(inFile)
    n_trials,n_subjects = fits.shape
    print('Done!')

    # Load learned params
    inFile = '%s/PyTorchParameters%s.csv'%(pytorchDir,suffix)
    print('Loading pyTorch parameters from %s...'%inFile)
    dfParams = pd.read_csv(inFile,index_col=0)
    # Clamp betaE and betaA to min 0, max 10
    dfParams.beta_E = np.clip(dfParams.beta_E,0,10)
    dfParams.beta_A = np.clip(dfParams.beta_A,0,10)
    n_trials,n_subjects = fits.shape
    print('Done!')

    # Load data
    if IS_EXPLORE:
        inFile = '%s/Mmi-gbeExplore_TrialForMdls.csv'%(pytorchDir)
    else:
        inFile = '%s/Mmi-GbeConfirm_TrialForMdls.csv'%(pytorchDir)
    # don't add late - we want to see how it fits data.
    print('Loading actual mood data from %s...'%inFile)
    dfData = pd.read_csv(inFile,index_col=0)
    mood = dfData['happySlider.response'].values.reshape((-1,n_trials)).T
    tMood = dfData['time'].values.reshape((-1,n_trials)).T
    print('Done!')

    # %% Calculate error
    MSE = np.nanmean((mood-fits)**2, axis=0)
    medianMSE = np.median(MSE)
    print('Median MSE = %.3g'%medianMSE)

    # %% Plot sample fits
    # Select random subjects reproducibly
    randSeed = 128
    np.random.seed(randSeed)
    iSubjToPlot = np.random.randint(low=0,high=mood.shape[1],size=3)
    nSubjToPlot = len(iSubjToPlot)
    # Set up figure
    plt.close(237)
    plt.figure(237,figsize=(12,6))
    plt.clf()
    paramNames = [r'$M_0$',r'$\lambda$',r'$\beta_E$',r'$\beta_A$',r'$\beta_T$',r'SSE']
    ylim2 = np.array([-.05,0.2])
    ylim1 = ylim2*10;
    # Plot
    for iPlot,iSubj in enumerate(iSubjToPlot):
        xBar = np.arange(dfParams.shape[1]-1)
        params = dfParams.iloc[iSubj,1:].values
        # plot big values
        color = 'tab:red'
        ax1 = plt.subplot(2,len(iSubjToPlot),iPlot+1)
        plt.bar(xBar[:2], params[:2],color=color,zorder=3)
        # annotate
        plt.ylabel('Parameter',color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        plt.ylim(ylim1)
        plt.grid(zorder=0)
        # plot small values
        color = 'tab:blue'
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        plt.bar(xBar[2:], params[2:],color=color,zorder=3)
        # annotate
        plt.ylabel('Parameter',color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        plt.ylim(ylim2)
        plt.grid(zorder=0)
        plt.xticks(xBar)
        plt.gca().set_xticklabels(paramNames)
        plt.title('Participant %d'%(iSubj))
        # Plot moods & fits
        plt.subplot(2,nSubjToPlot,iPlot+nSubjToPlot+1)
        plt.plot(tMood[:,iSubj],mood[:,iSubj],'.-',label='true mood')
        plt.plot(tMood[:,iSubj],fits[:,iSubj],'.-',label='model fit')
        # Annotate plot
        plt.ylim([0,1])
        plt.xlabel('Time (s)')
        plt.ylabel('Mood (0-1)')
        plt.grid(zorder=0)
        plt.legend()

    # Annotate figure
    plt.tight_layout()
    plt.savefig('%s/SampleFits%s.png'%(outFigDir,suffix))
    plt.savefig('%s/SampleFits%s.pdf'%(outFigDir,suffix))

    # %% Report mean r^2 across subjects
    for corrType in ['Spearman','Pearson']:

        # calculate r
        r = np.zeros(n_subjects)
        isRating = ~np.isnan(mood[:,0])
        if corrType=='Spearman':
            for i in range(n_subjects):
                r[i],_ = stats.spearmanr(mood[isRating,i],fits[isRating,i])
        else:
            for i in range(n_subjects):
                r[i],_ = stats.pearsonr(mood[isRating,i],fits[isRating,i])

        # transform to z
        r_z = np.arctanh(r)

        # replace infinity with next highest value
        if np.any(r==1):
            print('Replacing %d r=1 values with next highest value.'%np.sum(r==1))
            r_z[r==1] = np.nanmax(r_z[r<1])

        # get mean & CI
        mean_z = np.nanmean(r_z)
        alpha = 0.05
        #se = 1/np.sqrt(n_subjects*n_trials-3) # as if it were a single r value
        se = np.nanstd(r_z)/np.sqrt(np.sum(~np.isnan(r_z)))
        z = stats.norm.ppf(1-alpha/2)
        lo_z, hi_z = mean_z-z*se, mean_z+z*se
        mean_r,lo_r, hi_r = np.tanh((mean_z, lo_z, hi_z))

        print('Mean %s r (between mood & fits, mean across subjects) = %.3g, 95%% CI = %.3g, %.3g'%(corrType,mean_r,lo_r,hi_r))


    # %% Compare testing error of model with or without beta_T
    if IS_EXPLORE:
        if IS_LATE:
            suffix = '_tune-late'
            suffix_noBetaT = '_tune-late-NoBetaT'
        else:
            suffix = '_tune-Oct2020'
            suffix_noBetaT = '_tune-NoBetaT'
    else:
        if IS_LATE:
            suffix = '_tune-GbeConfirm-late'
            suffix_noBetaT = '_tune-GbeConfirm-late-NoBetaT'
        else:
            suffix = '_tune-GbeConfirm'
            suffix_noBetaT = '_tune-GbeConfirm-NoBetaT'


    nTestTrials = 3 # was 2, changed to 3 2/18/21
    MSE = np.load('%s/PyTorchTestLoss%s.npy'%(pytorchDir,suffix)) / nTestTrials
    MSE_noBetaT = np.load('%s/PyTorchTestLoss%s.npy'%(pytorchDir,suffix_noBetaT)) / nTestTrials
    median_losses = np.median(MSE, axis = 0) # Average loss across subjects as a function of regularization
    median_losses_noBetaT = np.median(MSE_noBetaT, axis = 0) # Average loss across subjects as a function of regularization

    if IS_EXPLORE:
        grid = np.load('%s/PyTorchPenaltyCoeff%s.npy'%(pytorchDir,suffix))
        grid_noBetaT = np.load('%s/PyTorchPenaltyCoeff%s.npy'%(pytorchDir,suffix_noBetaT))

        # loss vs. penalty is noisy - fit a kernel ridge regression
        Reg = KernelRidge(alpha=10.0, kernel='polynomial')
        Reg.fit(np.log(grid), median_losses)
        smoothed_losses = Reg.predict(np.log(grid))
        best_ind = np.argsort(smoothed_losses)[0]
        print('best penalty coeffs (WITH betaT): %s'%grid[best_ind, :])

        # Duplicate until we reach n_models
        Reg = KernelRidge(alpha=10.0, kernel='polynomial')
        Reg.fit(np.log(grid_noBetaT), median_losses_noBetaT)
        smoothed_losses_noBetaT = Reg.predict(np.log(grid_noBetaT))
        best_ind_noBetaT = np.argsort(smoothed_losses_noBetaT)[0]
        print('best penalty coeffs (NO betaT): %s'%grid_noBetaT[best_ind_noBetaT, :])


    print('min median MSE (WITH betaT): %.5g'%np.min(median_losses))
    print('min median MSE (NO betaT): %.5g'%np.min(median_losses_noBetaT))
    if IS_EXPLORE:
        diff = MSE[:,best_ind] - MSE_noBetaT[:,best_ind_noBetaT]
        median_best = np.median(MSE[:,best_ind])
        median_best_noBetaT = np.median(MSE_noBetaT[:,best_ind_noBetaT])
        stat,p = stats.wilcoxon(MSE[:,best_ind], MSE_noBetaT[:,best_ind_noBetaT])
    else:
        diff = median_losses - median_losses_noBetaT
        median_best = np.median(median_losses)
        median_best_noBetaT = np.median(median_losses_noBetaT)
        stat,p = stats.wilcoxon(median_losses, median_losses_noBetaT)
    IQR_diff = stats.iqr(diff)
    print(f'With vs NO betaT: median={np.median(median_best):.3g} vs. {np.median(median_best_noBetaT):.3g}, IQR={IQR_diff:.3g}')
    print(f'2-sided wilcoxon sign-rank test on losses with/without betaT for the median subject across regularizations: n={len(median_losses)}, dof={len(median_losses) - 1}, stat={stat}, p={p:.3g}')
