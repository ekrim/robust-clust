import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as ds
from sklearn.metrics import adjusted_rand_score as ari

import constrained_clustering as cc


Nclusters, N, Nconstraints = (3, 500, 10000)
data, labels = ds.make_blobs(n_samples=N,
			     n_features=2,
			     centers=Nclusters)

NconsVec = range(0,10001,1000)
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

