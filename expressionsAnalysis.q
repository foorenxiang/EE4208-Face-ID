\l p.q

p)from joblib import load
p)expressionSSDstore = load('expressionSSDstore.bin')
"expressionSSDstore"
show expressionSSDstore:.p.py2q .p.pyget `expressionSSDstore

\p 5001
\cd /Users/foorx/developer
\l launcher.q_
\cd /Users/foorx/opencv

show "Expressions"
show expressions:`EO`EF`EL`EC`NEUTRAL`ES

"EO expression euclidean distance from faces centroid"
show EO:expressionSSDstore[`EO]

show "EF expression euclidean distance from faces centroid"
show EF:expressionSSDstore[`EF]

show "EL expression euclidean distance from faces centroid"
show EL:expressionSSDstore[`EL]

show "EC expression euclidean distance from faces centroid"
show EC:expressionSSDstore[`EC]

show "neutral expression euclidean distance from faces centroid"
show neutral:expressionSSDstore[`neutral]

show "ES expression euclidean distance from faces centroid"
show ES:expressionSSDstore[`ES]

show "standard deviation of each expression"
eStdDev: dev each (EO;EF;EL;EC;neutral;ES)
show expressions!enlist each eStdDev

show "mean distance of each expression"
eMean: avg each (EO;EF;EL;EC;neutral;ES)
show expressions!enlist each eMean

"median distance of each expressions"
eMedian: med each (EO;EF;EL;EC;neutral;ES)
show expressions!enlist each eMedian

show "mean distance of mean of expressions"
show eMeanMean: avg eMean

show "distance bias of mean of each expression"
eMeanBias: eMean - eMeanMean
show expressions!enlist each eMeanBias

show "mean distance of median of each expression"
show eMedianMean: avg eMedian

show "distance bias of median of each expression"
eMedianBias: eMedian-eMedianMean
show expressions!enlist each eMedianBias

show "max distance of each expression"
eMax: max each (EO;EF;EL;EC;neutral;ES)
show expressions!enlist each eMax

show "min distance of each expression"
eMin: min each (EO;EF;EL;EC;neutral;ES)
show expressions!enlist each eMin

show "difference between max and min distance of each expression"
MaxMinDiff: eMax - eMin
show expressions!enlist each MaxMinDiff

show "smallest distance from expression centroid, for expression samples"
samplesToConsider:10
show closestToCentroid:expressions!{[expression] samplesToConsider# asc `int$abs expression - avg expression} each (EO;EF;EL;EC;neutral;ES)
show "indexes for above calculation"
show closestToCentroidIndices:{[expression] samplesToConsider# iasc `int$abs expression - avg expression} each (EO;EF;EL;EC;neutral;ES)