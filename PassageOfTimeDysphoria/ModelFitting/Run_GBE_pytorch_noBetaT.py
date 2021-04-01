#!/usr/bin/env python3
# coding: utf-8

# =========================
# Run_GBE_pytorch_noBetaT.py
# Pytorch code for fitting cognitive models
#
# Created 8/25/20 by Charles Zheng
# Updated 9/1/20 by DJ - moved plotting to a separate script
# Updated 9/9/20 by DJ - return & write mood predictions to file
# Updated 9/14/20 by DJ - converted times from seconds to minutes. and divided
#   outcomeAmount by 100, to avoid rounding errors. Stopped gradient update &
#   param change after final iteration, increased to 50k iterations.
# Updated 9/15/20 by DJ - removed one-iteration test section, increased n_models
#   to 40, switched to gpu, added command-line interface & out_suffix input
# Updated 9/16/20 by DJ - added out_folder input, saving objective function
#   every 100 iterations
# Updated 9/22/20 by DJ - added optimizer input
# Updated 9/23/20 by DJ - updated default learning rate (0.005), n_models (500),
#   and n_iterations (100k), optimizer (adam), and input file (Mmi-gbeExplore_TrialForMdls.csv)
# Updated 9/29/20 by DJ - updated default penalty terms (pen_EA=6.16, pen_T=2.64)
# Updated 10/1/20 by DJ - added participant column to best_pars table
# Updated 10/16/20 by DJ - changed default penalty hyperparameters
# Updated 10/19/20 by DJ - removed betaT
# Updated 10/20/20 by DJ - clamped mood_new to [0,1], changed default pen_betaE/A to 2.64
# Updated 10/21/20 by DJ - clamped final beta_A/E parameters to [0,10],
#   changed default pen_betaE/A to 0.483
# Updated 10/29/20 by DJ - commented out mood_rt
# =========================

# Import Python Packages
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import torch as pt # pt stands for PyTorch
import os # for folder detection & creation
import argparse
# Import functions for easy access
sg = pt.nn.Sigmoid()
from numpy.random import randn, rand
import torch.optim as optim


# ## Defining the model
#
# The LTA model with time drift is defined as follows
#
# $E(t) = \frac{1}{t-1}\sum_{u=1}^{t-1} A(u)$
#
# $\hat{M}(t) = M_0 + \beta_E \sum_{u=1}^t \lambda^{t-u} E(u) + \beta_A \sum_{u=1}^t \lambda^{t-u} A(u) + \beta_T T(t)$
#
# where $A(t)$ is the actual outcome of trial $t$, $T(t)$ is the time stamp, $M(t)$ is the mood rating.
#
# $\lambda$ and $M_0$ are constrained to lie in [0,1].  $\beta_A$ and $\beta_E$ are constrained to be nonnegative.  There are no constraints on $\beta_T$.
#
# ### Regularization
#
# We have an L2-norm penalty on $\beta_A, \beta_E, \beta_T$.  Hence the objective function per subject is
#
#
# $\text{minimize} \sum_{t \in \text{mood ratings}} (\hat{M}(t) - M(t))^2 + 0.1 * (\beta_A^2 + \beta_E^2 + \beta_T^2)$,
#
# where $M(t)$ is the mood rating at time $t$.
#
# ### Ensemble
#
# An individual subject model is defined by a parameter vector $\theta = (M_0, \lambda, \beta_E, \beta_A, \beta_T)$.  We fit not a single parameter vector but rather an _ensemble_ of models $\theta^1,..., \theta^m$.
# There are two uses of this:
#  * To check reproducibility and convergence wrt to multiple starting points
#  * To assess uncertainty.  To do this, one applies a penalty to the ensemble to ensure *diversity* of points in the ensemble.  This would allow us to see the uncertainty associated with each parameter fit. (However we do not use this approach here.)
#
# ### Fitting in parallel
#
# We run each subject in parallel.  For a given parameter (say, $M_0$), we store all ensembles for all subjects in the same PyTorch tensor.  Hence the dimension of the PyTorch tensor is (no. subjects, no. models).
#
# ### Initialization
#
# Parameters are initialized with sensible and randomized starting values.


# # Full procedure

def run_model(lte, outcomeAmount, mood_rating, time, pen_betaE, pen_betaA,
                learning_rate, optim_type, n_models, n_iterations, verbose,
                parameters = None):

    # Extract constants
    n_trials,n_subjects = lte.shape

    # Initialize parameters
    if parameters is None:
        dtype = pt.float
        # device = pt.device("cpu") # use CPU for PyTorch
        device = pt.device("cuda") # use GPU for PyTorch
        # M0 initialized to Normal(0, 1) (will be sigmoid-transformed to conform to [0, 1])
        par_M0 = pt.randn(n_subjects, n_models, device=device, dtype=dtype, requires_grad=True)
        par_lambda = pt.randn(n_subjects, n_models, device=device, dtype=dtype, requires_grad=True)
        # betaE, betaA initialized to Lognormal(0, 1)
        par_betaE = pt.tensor(np.exp(randn(n_subjects, n_models)), device=device, dtype=dtype, requires_grad=True)
        par_betaA = pt.tensor(np.exp(randn(n_subjects, n_models)), device=device, dtype=dtype, requires_grad=True)
        # Assemble all parameters
        parameters = [par_M0, par_lambda, par_betaE, par_betaA]
    par_M0, par_lambda, par_betaE, par_betaA = parameters

    # Set up objective function tracking
    obj_fn_track = []

    # optimizer
    if optim_type.lower()=='sgd':
        optimizer = optim.SGD(parameters, lr=learning_rate)
    elif optim_type.lower()=='adam':
        optimizer = optim.Adam(parameters, lr=learning_rate)
    else:
        raise ValueError('Optimizer type %s not recognized!'%optim_type)

    for iter_no in range(n_iterations):
        # holds the predicted moods for each of 30 trials, for all subjects and models simultaneously
        mood_predictions = []
        # Holds the exponentially weighted sums for E(t) and A(t)
        sum_E = [pt.zeros(n_subjects, n_models, device=device, dtype=dtype)]
        sum_A = [pt.zeros(n_subjects, n_models, device=device, dtype=dtype)]
        # holds mood losses in absolute error
        mood_losses = []

        for trial_no in range(n_trials):
            ## Clamping variables
            pl = sg(par_lambda)
            m0 = sg(par_M0)
            pbA = pt.clamp(par_betaA, 0.0, 10.0)
            pbE = pt.clamp(par_betaE, 0.0, 10.0)
            # ------------------------
            # Computing predicted mood
            # ------------------------
            sum_E_new = sum_E[trial_no] * pl.reshape((n_subjects, n_models)) +                 pt.tensor(lte[trial_no, :].reshape(n_subjects, 1), device=device, dtype=dtype)
            sum_A_new = sum_A[trial_no] * pl.reshape((n_subjects, n_models)) +                 pt.tensor(outcomeAmount[trial_no, :].reshape(n_subjects, 1), device=device, dtype=dtype)
            sum_E.append(sum_E_new)
            sum_A.append(sum_A_new)
            mood_new = m0 + pbA * sum_A_new + pbE * sum_E_new
            mood_new = pt.clamp(mood_new, 0.0, 1.0)
            mood_predictions.append(mood_new)
            # -------------------------
            # Computing losses for mood
            # -------------------------
            true_mood = mood_rating[trial_no, :]
            if np.all(~np.isnan(true_mood)):
                true_mood_new = pt.tensor(true_mood.reshape(n_subjects, 1), device=device, dtype=dtype)
                # loss_new = pt.abs(true_mood_new - mood_new) # for SAE
                loss_new = pt.square(true_mood_new - mood_new) # for SSE
                mood_losses.append(loss_new)


        # compute regularization terms
        reg_pen = pen_betaE * pt.sum(par_betaE**2) +                     pen_betaA * pt.sum(par_betaA**2)
        total_loss = pt.sum(sum(mood_losses)) + reg_pen

        # Save out losses every 100 iterations
        if (iter_no % 100) == 0:
            obj_fn_track.append(total_loss.item())

        # compute gradient
        if iter_no<(n_iterations-1):
            optimizer.zero_grad()   # zero the gradient buffers
            total_loss.backward(retain_graph=True)
            optimizer.step()    # Does the update

        # shows the losses
        if verbose:
            if (iter_no % int(verbose)) == 0:
                vv = sum(mood_losses).data
                print('iter %d/%d...'%(iter_no,n_iterations))
                print(vv)

    return parameters, sum(mood_losses), mood_predictions, obj_fn_track




# ======== MAIN FUNCTION ========= #

def LoadDataAndRunModel(data_file='Mmi-gbeExplore_TrialForMdls.csv', n_subjects=5000, n_trials = 30,
                        pen_betaE=0.1, pen_betaA=0.1,
                        learning_rate = 0.005, optim_type = 'adam', n_models = 500,
                        n_iterations = 100000, out_folder = './', out_suffix=''):

    # Make output directory
    if not os.path.exists(out_folder):
        print('=== Making output directory %s... ==='%out_folder)
        os.makedirs(out_folder)

    # Import the data
    print('=== Loading Data... ===')
    ts = pd.read_csv(data_file).iloc[:(n_subjects*n_trials), :] # subset of first 100 subjects for development purposes

    # Print columns
    print('Columns:')
    print(ts.columns)

    # reformat data into separate variables that are #trials x #subjects
    trial_no = ts.trial_no.values.reshape((-1, n_trials)).T
    participant = ts.participant.values.reshape((-1, n_trials)).T
    time = ts.time.values.reshape((-1, n_trials)).T / 60.0 # Time in minutes
    # winAmount = ts.winAmount.values.reshape((-1, n_trials)).T
    # loseAmount = ts.loseAmount.values.reshape((-1, n_trials)).T
    # certainAmount = ts.certainAmount.values.reshape((-1, n_trials)).T
    # choice = ts.choice.map({'gamble': 1, 'certain': 0}).values.reshape((-1, n_trials)).T
    outcomeAmount = ts.outcomeAmount.values.reshape((-1, n_trials)).T / 100.0 # TEST
    # choiceKey_rt = ts['choiceKey.rt'].values.reshape((-1, n_trials)).T
    mood_rating = ts['happySlider.response'].values.reshape((-1, n_trials)).T
    # mood_rt = ts['happySlider.rt'].values.reshape((-1, n_trials)).T

    # compute LTEs
    temp = np.cumsum(outcomeAmount, axis=0)/(trial_no+1)
    lte = np.zeros((n_trials, n_subjects))
    lte[1:, :] = temp[:-1, :]

    # RUN THE MODEL
    print('=== Running Model... ===')
    pars, losses, mood_pred, obj_fn = run_model(lte=lte, outcomeAmount=outcomeAmount, mood_rating=mood_rating, time=time,
              pen_betaE = pen_betaE, pen_betaA = pen_betaA,
              learning_rate = learning_rate, optim_type = optim_type,
              n_models = n_models, n_iterations = n_iterations, verbose=2000)


    # Pull out info from output
    print('=== Extracting Results... ===')
    sols = list([p.cpu().detach().numpy() for p in pars])
    losses_ = losses.cpu().detach().numpy()
    mood_pred_3d = np.array(list([m.cpu().detach().numpy() for m in mood_pred]))
    obj_fn_np = np.array(obj_fn)

    # get parameters & mood predictions from best-fitting model
    parnames = ['m0', 'lambda', 'beta_E', 'beta_A']
    best_pars = pd.DataFrame(np.zeros((n_subjects,len(parnames)+1)), columns=['participant']+parnames)
    best_pars['participant'] = participant[0,:].T # add participant labels
    best_mood_pred = np.zeros(mood_pred_3d.shape[:2]) # mood predictions from best model
    for sub_no in range(n_subjects):
        best_ind = np.argmin(losses_[sub_no, :])
        for par_no in range(len(parnames)):
            best_pars.loc[sub_no, parnames[par_no]] = sols[par_no][sub_no, best_ind]
            best_pars.beta_A = np.clip(best_pars.beta_A,0,10)
            best_pars.beta_E = np.clip(best_pars.beta_E,0,10)
            best_pars.loc[sub_no, 'SSE'] = losses_[sub_no, best_ind]
        # Get mood predictions from this best model
        best_mood_pred[:,sub_no] =  mood_pred_3d[:,sub_no,best_ind]

    # Apply logistic function to lambda & m0
    best_pars['lambda'] = 1.0/(1.0 + np.exp(-best_pars['lambda']))
    best_pars['m0'] = 1.0/(1.0 + np.exp(-best_pars['m0']))

    # Print best parameters
    print('Parameters:')
    print(best_pars)

    # Save results
    print('=== Saving Results... ===')
    out_file = '%s/PyTorchParameters%s.csv'%(out_folder,out_suffix)
    print('Saving best parameters to %s...'%out_file)
    best_pars.to_csv(out_file);
    print('Done!')

    out_file = '%s/PyTorchPredictions%s.npy'%(out_folder,out_suffix)
    print('Saving model predictions to %s...'%out_file)
    np.save(out_file,best_mood_pred)
    print('Done!')

    out_file = '%s/PyTorchObjectiveFn%s.npy'%(out_folder,out_suffix)
    print('Saving objective function to %s...'%out_file)
    np.save(out_file,obj_fn_np)
    print('Done!')




# %% Make command-line argument parser
parser = argparse.ArgumentParser(description="Train Charles Zheng's LTA model with time drift on GBE data using PyTorch and a GPU.")
# Add arguments
parser.add_argument('-df','--data_file', default='Mmi-gbeExplore_TrialForMdls.csv', help='filename of input data file')
parser.add_argument('-ns','--n_subjects', default='5000', help='number of subjects to include')
parser.add_argument('-nt','--n_trials', default='30', help='number of trials per subject')
parser.add_argument('-pe','--pen_betaE', default='0.483', help='regularization penalty on beta_E parameter')
parser.add_argument('-pa','--pen_betaA', default='0.483', help='regularization penalty on beta_A parameter')
parser.add_argument('-lr','--learning_rate', default='0.005', help='learning rate')
parser.add_argument('-nm','--n_models', default='500', help='number of models to train and choose from')
parser.add_argument('-ni','--n_iterations', default='100000', help='number of learning iterations to run on each model')
parser.add_argument('-ot','--optim_type', default='adam', help='optimizer type (adam or sgd)')
parser.add_argument('-of','--out_folder', default='./', help='folder in which to place output files')
parser.add_argument('-os','--out_suffix', default='', help='suffix at end of output files')

# %% ==== COMMAND-LINE MAIN FUNCTION ====

if __name__ == '__main__':

    # Parse inputs
    args = parser.parse_args();

    # Call main function
    LoadDataAndRunModel(data_file = args.data_file, n_subjects = int(args.n_subjects), n_trials = int(args.n_trials),
                        pen_betaE = float(args.pen_betaE), pen_betaA = float(args.pen_betaA),
                        learning_rate = float(args.learning_rate), optim_type = args.optim_type, n_models = int(args.n_models),
                        n_iterations = int(args.n_iterations), out_folder = args.out_folder, out_suffix=args.out_suffix)
