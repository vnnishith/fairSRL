isNotPrivileged(X):- X \= 'male'.
overrideProbability(M,N):- M > N.
fairModel(Sex,Prob,P) :- isNotPrivileged(Sex) -> (overrideProbability(Prob,0.25) ->  P is 1 ; P is 0); overrideProbability(Prob,0.60) ->  P is 1 ; P is 0.
