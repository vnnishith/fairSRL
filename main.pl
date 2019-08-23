isNotPrivileged(X):- X \= 'white'.
overrideProbability(M,N):- M > N.
fairModel(Race,Prob,P) :- isNotPrivileged(Race) -> (overrideProbability(Prob,0.3) ->  P is 1 ; P is 0); overrideProbability(Prob,0.5) ->  P is 1 ; P is 0.
