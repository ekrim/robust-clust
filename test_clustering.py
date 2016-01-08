import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as ds

import constrained_clustering as cc


Nclusters, N, Nconstraints = (3, 1000, 10000)
data, labels = ds.make_blobs(n_samples=N,
			     n_features=2,
			     centers=Nclusters)

constraintMat = cc.ConstrainedClustering.make_constraints(labels, Nconstraints=Nconstraints)

specLearning = cc.SpectralLearning(data=data, constraintMat=constraintMat, n_clusters=Nclusters)

specLearning.fit_constrained()

plt.figure()
cc.plot_labels(data)
cc.plot_labels(data, specLearning.labels)
plt.show()

