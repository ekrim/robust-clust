import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as ds
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.neighbors import KNeighborsClassifier

import constrained_clustering as cc
import constraints_to_labels as ctl

Nclusters, N, Nconstraints = (3, 500, 500)
data, labels = ds.make_blobs(n_samples=N,
			     n_features=2,
			     centers=Nclusters)

"""NconsVec = range(0,10001,1000)
ariVec = []
for Ncons in NconsVec:
	constraintMat = cc.ConstrainedClustering.make_constraints(labels, Nconstraints=Nconstraints)
	
	specLearning = cc.SpectralLearning(data=data, constraintMat=constraintMat, n_clusters=Nclusters)
	specLearning.fit_constrained()
	
	ariVec += [ari(labels, specLearning.labels)]


plt.figure()
cc.plot_labels(data)
cc.plot_labels(data, specLearning.labels)
plt.show()

plt.figure()
plt.plot(NconsVec, ariVec, '-')
plt.show()
"""

constraintMat = cc.ConstrainedClustering.make_constraints(labels, Nconstraints=Nconstraints)

e2cp = cc.E2CP(data=data, constraintMat=constraintMat, n_clusters=Nclusters)
e2cp.fit_constrained()
ariE2CP = ari(labels, e2cp.labels)

"""plt.figure()
cc.plot_labels(data)
cc.plot_labels(data, e2cp.labels)
plt.show()
"""

ctlObj = ctl.ConstraintsToLabels(data=data, 
		constraintMat=constraintMat, 
		n_clusters=None)
ctlObj.fit_constrained()

trainLabels = ctlObj.labelSet
trainInd = ctlObj.constrainedSamps
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(data[trainInd,:], labels[trainInd])
ctlLabels = neigh.predict(data)
ariCTL = ari(labels, ctlLabels)

print ariE2CP, ariCTL



