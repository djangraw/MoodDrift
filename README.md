# Passage-of-Time Dysphoria

This repository contains the scripts necessary to preprocess, analyze, print and plot the results reported in the following paper:

> Jangraw, Keren, Bedder, Rutledge, Pereira et al. (2021). "Passage-of-Time Dysphoria: A Highly Replicable, Gradual Decline in Subjective Mood Observed During Rest and Simple Tasks." BioRxiv.

Analyses are performed on the data found in the following repository on the Open Science Framework: https://osf.io/km69z/.

## Dependencies
- pymer4, available at https://eshinjolly.com/pymer4
### Fitting models to GBE data
- pytorch, available at https://pytorch.org

Code for preprocessing, fitting mixed effects models, and analysis and plotting of results will work from a conda environment created with this command:
```
conda create -p potdys3 -c ejolly -c defaults -c conda-forge python=3.8.8 numpy=1.19.2 pandas=1.2.3 pytest=6.2.2 joblib=1.0.1 rpy2=3.4.3 r-base=3.6.3 r-lme4=1.1_21 pymer4=0.7.1 matplotlib=3.3.4 seaborn=0.11.1
```

If you need an environment for running the pytorch models, please open an issue and we'll work on adding this.


## Usage
Data can be downloaded to the Data folder from https://osf.io/km69z/ . Figures will be saved to the Figures folder.
We have broken down processing into 3 processing folders that can be executed sequentially: Preprocessing, Model Fitting, and Analysis. Each is found in its own folder, but some depend on scripts in other folders, so all three should be added to your Python path.

### Preprocessing
Converts raw data into preprocessed files that combine across cohorts (or "batches") of participants. Run the following:
- [ImportMmiData_May2020.py](Preprocessing/ImportMmiData_May2020.py) to import online adult cohorts
- [ImportNimhMmiData.py](Preprocessing/ImportNimhMmiData.py) to import online adolescent cohorts
- [AssembleMmiBatches.py](Preprocessing/AssembleMmiBatches.py) to combine across cohorts for grouped analyses
- [MakePytorchInputTable.py](Preprocessing/MakePytorchInputTable.py) to format data for computational model fitting.

All other scripts and functions in this folder will be called by these parent scripts.

### Model Fitting
Fits the large-scale linear mixed effects (LME) model and the computational model described in the paper. The computational model scripts are computationally expensive and best run on a high-performance computing cluster. Run the following:

- [RunPymerOnCovidData_Aug2020.py](ModelFitting/RunPymerOnCovidData_Aug2020.py) to run the LME model. Must be run from a python environment with pymer4 installed.
- [Tune_GBE_pytorch.py](ModelFitting/Tune_GBE_pytorch.py) to tune the computational model's hyperparameters. Must have pytorch installed.
- [Tune_GBE_pytorch_NoBetaT.py](ModelFitting/Tune_GBE_pytorch_NoBetaT.py) to do the same without the time-responsive beta_T term.
- [Tune_GBE_pytorch.py](ModelFitting/Tune_GBE_pytorch.py) to run the model with given hyperparameters. Must have pytorch installed.
- [Run_GBE_pytorch_NoBetaT.py](ModelFitting/Tune_GBE_pytorch_NoBetaT.py) to do the same without the time-responsive beta_T term.
- [CombineGbeConfirmResults.py](ModelFitting/CombineGbeConfirmResults.py) to combine across computational model results for the confirmatory mobile app sub-cohorts, which were split apart to avoid memory errors.

### Analysis
Run [ProduceAllResults.py](Analysis/ProduceAllResults.py). All other scripts and functions in this folder will be called by this wrapper script.
