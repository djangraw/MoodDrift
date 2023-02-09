# Mood Drift Over Time Data

The data in this repository are presented in the following paper:
> Jangraw et al. (2023). "A Highly Replicable Decline in Mood During Rest and Simple Tasks." Nature Human Behaviour.

- The DataCheck folder contains .csv files containing links to each raw data file.
- The PilotData folder contains one subfolder for each of the batches. The raw data files for all participants in that batch sit inside that subfolder.
- The OutFiles folder contains the resulting processed data files combining across the participants in each batch.
- OutFiles/Mmi-Batches.csv contains information about each batch and the path to each of its raw data files.

## Analysis
The scripts used to analyze these data can be found on GitHub at
[github.com/djangraw/MoodDrift](github.com/djangraw/MoodDrift)

## Naming Conventions

Cohorts were renamed after analysis to make them more intuitive in the paper. Here is a translation:

| Name in paper | Folder(s) in database |
| ------------- | ------------------ |
| 15sRestBetween | Recovery1, RecoveryInstructed1 |
| 30sRestBetween | RecoveryInstructed1Freq0p5 |
| 7.5sRestBetween | RecoveryInstructed1Freq2 |
| 60sRestBetween | RecoveryInstructed1Freq0p25 |
| AlternateRating | Numbers |
| Expectation-7mRest | Expectation-7min |
| Expectation-12mRest | Expectation-12min |
| RestDownUp | RestDownUp |
| Daily-Rest-01 | Stability01-Rest |
| Daily-Rest-02 | Stability02-Rest |
| Weekly-Rest-01 | COVID01 |
| Weekly-Rest-02 | COVID02 |
| Weekly-Rest-03 | COVID03 |
| Adolescent-01 | RecoveryNimh |
| Adolescent-02 | RecoveryNimh |
| Visuomotor | Motion |
| Visuomotor-Feedback | MotionFeedback |
| RestAfterWins | Return1 |
| Daily-Closed-01 | Stability01-closed |
| Daily-Random-01 | Stability02-RandomVer2 |
| Activities | Activities |
| BoredomBeforeAndAfter | BoredomBeforeAndAfter |
| BoredomAfterOnly | BoredomAfterOnly |
| MwBeforeAndAfter | MwBeforeAndAfter |
| MwAfterOnly| MwAfterOnly |
| App-Exploratory | *produced using [Dryad data](https://doi.org/10.5061/dryad.prr4xgxkk)* |
| App-Confirmatory | *produced using [Dryad data](https://doi.org/10.5061/dryad.prr4xgxkk)* |
