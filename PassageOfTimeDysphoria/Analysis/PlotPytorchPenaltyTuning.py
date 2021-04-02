#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PlotPytorchPenaltyTuning.py
Created on Mon Sep 28 14:39:05 2020

@author: jangrawdc

- Updated 10/29/20 by DJ - fixed axis labels & titles, incl. capitalization
- Updated 3/31/21 by DJ - adapted for shared code structure.
"""

# %%
import numpy as np
from matplotlib import pyplot as plt
from sklearn.kernel_ridge import KernelRidge

def PlotPenaltyTuning(suffix,dataDir='../Data/OutFiles',outFigDir='../Figures'):

    grid = np.load('%s/PyTorchPenaltyCoeff%s.npy'%(dataDir,suffix))
    tl = np.load('%s/PyTorchTestLoss%s.npy'%(dataDir,suffix))
    nSubj,nModels = tl.shape

    # loss vs. penalty is noisy - fit a kernel ridge regression
    median_losses = np.median(tl, axis = 0) # Average loss across participants as a function of regularization
    Reg = KernelRidge(alpha=10.0, kernel='polynomial')
    Reg.fit(np.log(grid), median_losses)
    smoothed_losses = Reg.predict(np.log(grid))
    best_ind = np.argsort(smoothed_losses)[0]
    print('best penalty coeffs: %s'%grid[best_ind, :])

    plt.figure(243,figsize=(10,5)); plt.clf()
    plt.subplot(1,3,1)
    plt.imshow(tl)
    plt.clim(0,1)
    plt.colorbar()
    plt.xlabel('Model')
    plt.ylabel('Participant')
    plt.title('Testing loss')

    # Plot loss for pen_betaEA
    plt.subplot(2,3,2)
    plt.semilogx(grid[:,0], median_losses,'.');#,c=np.log(grid[:,1]))
    plt.gca().set_xscale('log')
    #plt.colorbar()
    plt.xlabel(r'$\lambda_{EA}$')
    plt.ylabel('Median losses\nacross %d participants'%nSubj)

    # Plot smoothed loss for pen_betaEA
    plt.subplot(2,3,5)
    plt.semilogx(grid[:,0], smoothed_losses,'.');#,c=np.log(grid[:,1]))
    #plt.gca().set_xscale('log')
    #plt.colorbar()
    plt.xlabel(r'$\lambda_{EA}$')
    plt.ylabel('Smoothed losses\nacross %d participants'%nSubj)


    if grid.shape[1]>1:
        # Plot loss for pen_betaT
        plt.subplot(2,3,3)
        plt.semilogx(grid[:,1], median_losses,'.');#,c=np.log(grid[:,1]))
        plt.gca().set_xscale('log')
        #plt.colorbar()
        plt.xlabel(r'$\lambda_{T}$')
        plt.ylabel('Median losses\nacross %d participants'%nSubj)

        # Plot smoothed loss for pen_betaT
        plt.subplot(2,3,6)
        plt.semilogx(grid[:,1], smoothed_losses,'.');#,c=np.log(grid[:,1]))
        plt.gca().set_xscale('log')
        #plt.colorbar()
        plt.xlabel(r'$\lambda_{T}$')
        plt.ylabel('Smoothed losses\nacross %d participants'%nSubj)
    else:
        print('grid does not have 2 columns... skipping second column.')



    plt.tight_layout(rect=(0,0,1.0,0.94))
    plt.suptitle('PyTorch Penalty Hyperparameter Tuning')
    plt.savefig('%s/PyTorch_PenaltyTuning%s.png'%(outFigDir,suffix))

    # %% Plot loss vs. pen_T at best pen_EA

    if 'noBetaT' not in suffix:
        plt.figure(266); plt.clf();
        isBestEA = grid[:,0]==grid[best_ind,0]
        plt.semilogx(grid[isBestEA,1],median_losses[isBestEA],label='median')
        plt.semilogx(grid[isBestEA,1],smoothed_losses[isBestEA],label='smoothed')
        plt.legend()
        plt.xlabel(r'$\lambda_{T}$')
        plt.ylabel('Median losses\nacross %d participants'%nSubj)
        plt.title(r'Losses at $\lambda_{EA}=%g$'%grid[best_ind,0])
        plt.tight_layout()
        plt.savefig('%s/PyTorch_PenaltyTuning_betaT%s.png'%(outFigDir,suffix))
