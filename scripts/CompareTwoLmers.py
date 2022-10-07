#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CompareTwoLmers.py

Compare the within- and between-subject variance explained by two LME models of the same data.

- Created 8/26/22 by DJ.
- Updated 9/30/22 by DJ - comments.
"""

# Import
import numpy as np
import pandas as pd
# Import pymer functions
from pymer4.models import Lmer
from rpy2 import robjects
from scipy import stats
mumin = robjects.r('library(MuMIn)') # for F-test
#vif = robjects.r('library(car)') # for VIF test

# %% Declare functions
# note, this variance does not include residual
# ripped off from mumin
get_varRE = robjects.r('''
function (object){
    fe <- MuMIn:::.numfixef(object)
    ok <- !is.na(fe)
    fitted <- (model.matrix(object)[, ok, drop = FALSE] %*% fe[ok])[,
        1L]
    varFE <- var(fitted)
    mmRE <- MuMIn:::.remodmat(object)
    vc <- MuMIn:::.varcorr(object)
    for (i in seq.int(length(vc))) {
        a <- MuMIn:::fixCoefNames(rownames(vc[[i]]))
        dimnames(vc[[i]]) <- list(a, a)
    }
    colnames(mmRE) <- MuMIn:::fixCoefNames(colnames(mmRE))
    if (!all(unlist(sapply(vc, rownames), use.names = FALSE) %in%
        colnames(mmRE)))
        stop("RE term names do not match those in model matrix. \n",
            "Have 'options(contrasts)' changed since the model was fitted?")
    varRE <- MuMIn:::.varRESum(vc, mmRE)
}
''')

def r_squaredSB(model, model_null, group_by = 'Subject'):
    #based on description here:
    # https://besjournals.onlinelibrary.wiley.com/doi/10.1111/j.2041-210x.2012.00261.x
    # variance of the random effects calculated taking into account correlation
    vre = get_varRE(model.model_obj)[0]
    # variance of the residual
    vee = model.ranef_var.loc['Residual', 'Var']

    # harmonic mean of the number of samples per subject
    k = stats.hmean(model.data.groupby(group_by).Time.nunique().values)

    # variances of the null model
    vre_null = get_varRE(model_null.model_obj)[0]
    vee_null = model_null.ranef_var.loc['Residual', 'Var']

    r_squared1 = 1 - ((vee + vre) / (vee_null + vre_null))

    r_squared2 = 1 - ((vee + (vre / k)) / (vee_null + (vre_null / k)))

    return r_squared1, r_squared2



# LMER comparison function from Dylan
def compare_lmers(pymer_input, lm_string_a, lm_string_ab, lm_string_null):

    """
    Run an anova comparing to mixed effects models on the same data.
    Note that the mixed effects structrure for both models should be the same.
    Also calculate r-squared and f-squared based on marginal r-squared.

    Parameters
    ----------
    pymer_input : Pandas.DataFrame
        Data to run the models on
    lm_string_a : str
        lme4 model string for the reduced model
    lm_string_ab : str
        lme4 model string for the more complex model
    lm_string_null : str
        lme4 model string for the null model common to the two above,
        should include all random effects, but probably just an intercept for fixed effects

    Returns
    -------
    anova_res : Pandas.DataFrame
        Results of the anova including marginal and conditional r-squared and f-squared.

    References
    ----------
    Formula for f-squared:
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3328081/#:~:text=Cohen's%20f%202%20(Cohen%2C%201988,R%202%201%20%2D%20R%202%20.

    Where I found the r-squared for mixed effects models function:
    https://stats.stackexchange.com/a/347908
    """
    # fit more complex LME model to to the data
    print('*** Fitting more complex model... ***')
    model_ab = Lmer(lm_string_ab, data=pymer_input)
    _ = model_ab.fit(REML=False, old_optimizer=True)
    dfFit_ab = model_ab.coefs
    dfFixef_ab = model_ab.fixef
    # fit reduced complex LME model to to the data
    print('*** Fitting reduced model... ***')
    model_a = Lmer(lm_string_a, data=pymer_input)
    _ = model_a.fit(REML=False, old_optimizer=True)
    dfFit_a = model_a.coefs
    dfFixef_a = model_a.fixef
    # fit null LME model to to the data
    print('*** Fitting null model... ***')
    model_null = Lmer(lm_string_null, data=pymer_input)
    _ = model_null.fit(REML=False, old_optimizer=True)
    # run anova to compare variance explained
    anova_res = pd.DataFrame((robjects.r('anova')(model_a.model_obj, model_ab.model_obj, refit=False)))
    # double-check that reduced model is actually smaller
    if len(dfFit_ab) <= (len(dfFit_a)):
        raise ValueError("Model AB should be a larger model than Model A, but based on the fixed effects, that's not the case.")
    # get rsquared values
    rsquaredab = robjects.r('r.squaredGLMM')(model_ab.model_obj)[0]
    rsquareda = robjects.r('r.squaredGLMM')(model_a.model_obj)[0]
    r2lr = robjects.r('''
    function (object)
    {
        r.squaredLR(object, null.RE = TRUE, adj.r.squared=TRUE)
    }
    ''')
    rsquaredab_lr = r2lr(model_ab.model_obj)[0]
    rsquareda_lr = r2lr(model_a.model_obj)[0]
    # get groupings
    group_by = lm_string_null.split('|')[1][:-1] # get null model random-effects factor, remove trailing )
    print(f'Grouping by {group_by}')
    # calculate R1^2 and R2^2 using SB method
    rsquaredab1, rsquaredab2  = r_squaredSB(model_ab, model_null, group_by)
    rsquareda1, rsquareda2  = r_squaredSB(model_a, model_null, group_by)

    # add rsquared and F-test results to anova table
    anova_res['marginal_R2'] = (rsquareda[0], rsquaredab[0])
    anova_res['conditional_R2'] = (rsquareda[1], rsquaredab[1])
    anova_res['lr_marginal_R2'] = (rsquareda_lr, rsquaredab_lr)
    anova_res['withinsubject_R2'] = (rsquareda1, rsquaredab1)
    anova_res['betweensubject_R2'] = (rsquareda2, rsquaredab2)

    anova_res['f2'] = (np.nan, (rsquaredab[0] - rsquareda[0]) / (1 - (rsquaredab[0])))
    anova_res['lr_f2'] = (np.nan, (rsquaredab_lr - rsquareda_lr) / (1 - (rsquaredab_lr)))
    anova_res['ws_f2'] = (np.nan, (rsquaredab1 - rsquareda1) / (1 - (rsquaredab1)))
    anova_res['bs_f2'] = (np.nan, (rsquaredab2 - rsquareda2) / (1 - (rsquaredab2)))


    #dfFit_a['VIF'] = np.insert(robjects.r('vif')(model_a.model_obj),0,0)
    #dfFit_ab['VIF'] = np.insert(robjects.r('vif')(model_ab.model_obj),0,0)
    # return results
    return anova_res, dfFit_a, dfFit_ab, dfFixef_a

def print_comparison_results(batch,pymer_input,lm_string_h0,lm_string_h1,anova_res,dfFit_h0,dfFit_h1):
    # get number of groups
    group_by = lm_string_h0.split('|')[1][:-1] # get null model random-effects factor, remove trailing )
    print(f'Grouping by {group_by}')
    group_count = len(np.unique(pymer_input[group_by]))

    # Print results
    print('=======BATCH=========')
    print(f'batch = {batch}')
    print(f'rows: {pymer_input.shape[0]}')
    print(f'groupings: {group_count}')
    print('=======MODELS=========')
    print(f'H0: {lm_string_h0}')
    print(f'H1: {lm_string_h1}')
    print('=======ANOVA=========')
    print(anova_res)
    print('=======FIT_H0=========')
    print(dfFit_h0)
    print('=======FIT_H1=========')
    print(dfFit_h1)
    print('================')

def compare_lmers_wrapper(batch='AllOpeningRestAndRandom',results_dir='../Data/OutFiles'):
    # load pymer input table
    in_file = f'{results_dir}/Mmi-{batch}_pymerInput-full.csv'
    print(f'Opening {in_file}...')
    pymer_input = pd.read_csv(in_file, index_col=0)

    # Declare model strings
    lm_string_h0 = 'Mood ~ 1 + fracRiskScore + Time * (isMale + meanIRIOver20 + isAge40to100) + (1 + Time|Subject)'
    # lm_string_h0 = 'Mood ~ 1 + fracRiskScore + isMale + meanIRIOver20 + isAge40to100 + (1 + Time|Subject)'
    lm_string_h1 = 'Mood ~ 1 + Time * (fracRiskScore + isMale + meanIRIOver20 + isAge40to100) + (1 + Time|Subject)'
    # lm_string_h0 = 'Mood ~ 1 + (1 + Time|Subject)'
    # lm_string_h1 = 'Mood ~ 1 + Time + (1 + Time|Subject)'

    lm_string_null = 'Mood ~ 1 + (1 + Time|Subject)'

    # d = difference between the means / ( sqrt( var.intercept_part + var.intercept_item + var.slope_part + var.slope_item + var_residual ) )*'



    # Run ANOVA
    anova_res, dfFit_h0, dfFit_h1, dfFixef_h0 = compare_lmers(pymer_input,lm_string_h0,lm_string_h1, lm_string_null)

    # Print results
    print_comparison_results(batch,pymer_input,lm_string_h0,lm_string_h1,anova_res,dfFit_h0,dfFit_h1)


# %% Run it
if __name__ == "__main__":
    results_dir = '../Data/OutFiles'
    batch = 'AllOpeningRestAndRandom'
    compare_lmers_wrapper(batch,results_dir)
