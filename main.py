import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as ds

data,labels = ds.make_blobs(n_samples=100,n_features=2,centers=3)

classes = set(labels)
colors = plt.cm.Spectral(np.linspace(0,1,len(classes)))

for lab,col in zip(classes,colors):
	ind = labels == lab
	plt.plot(data[ind,0],data[ind,1],'o',markerfacecolor=col,markersize=10)

plt.show()
