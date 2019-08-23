# FairSRL framework
* Python version 3.4
* The app can be started by running python3 main.py.
* Required Dependencies: matplotlib, scikit-learn, Tkinter, pandas, pyswip.

## Test Datasets
### Adult Census Dataset
* https://archive.ics.uci.edu/ml/datasets/adult

### Loan Prediction Dataset
* https://datahack.analyticsvidhya.com/contest/practice-problem-loan-prediction-iii/.

<br/>
<br/>

### Fairness and rules
* The different fairness metrics considered are proportion and statistical parity difference between different groups.
* FairSRL ensure equal opportunity to groups by using a logic model in conjunction with a machine learning model to ensure fairness.
* FairSRL allows custom thresholds to different groups as well as support to add custom rules to provide more opportunities to underprivileged group.
* The rules are to be written in [Prolog](https://staff.fnwi.uva.nl/u.endriss/teaching/prolog/prolog.pdf).
* Sample example for rule being used for the logic model.
```pearl
> isNotPrivileged(X):- X \= 'white'.
> overrideProbability(M,N):- M > N.
> fairModel(Race,Prob,P) :- isNotPrivileged(Race) -> (overrideProbability(Prob,0.3) ->  P is 1 ; P is 0); overrideProbability(Prob,0.5) ->  P is 1 ; P is 0.
```



