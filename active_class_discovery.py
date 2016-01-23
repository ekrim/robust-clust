import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as ds
from sklearn.cluster import AgglomerativeClustering

import constrained_clustering as cc


class ActiveClassDiscovery(object):
	def __init__(self, data, knownInd=None):
		self.data = data
		self.N = data.shape[0]
		self.hierarchical = AgglomerativeClustering(linkage='average')
		self.hierarchical.fit(data)
		self.linkage_to_merges()
		
		self.knownLabels = -np.ones(self.N)
		if knownInd is not None:
			self.knownLabels[knownInd] = 1			

	def get_query(self):		
		numberOfEmpties = np.zeros((self.N-1,2))
		for i in range(self.N-1):
			ind1 = self.mergeHistory[i][0]
			ind2 = self.mergeHistory[i][1]
			
			lab1 = np.unique(self.knownLabels[ind1])
			lab2 = np.unique(self.knownLabels[ind2])
			if not np.any(lab1 != -1):
				numberOfEmpties[i,0] = ind1.size
			if not np.any(lab2 != -1):
				numberOfEmpties[i,1] = ind2.size	
		bestSides = np.argmax(numberOfEmpties, axis=1)
		biggestEmpty = np.argmax(numberOfEmpties[np.arange(self.N-1),bestSides])		
		emptySide = bestSides[biggestEmpty]
		
		queryGroup = self.mergeHistory[biggestEmpty][emptySide] 
		bestSample = np.random.random_integers(0,queryGroup.size-1)
		newQuery = queryGroup[bestSample]
		self.knownLabels[newQuery] = 1
		return newQuery	

	def linkage_to_merges(self):
		self.mergeHistory = []
		clusMem = [np.asarray([x]) for x in range(self.N)]
		for i in range(0, self.N-1):
			group1 = self.hierarchical.children_[i,0]
			group2 = self.hierarchical.children_[i,1]
			clusMem += [np.append(clusMem[group1], clusMem[group2])]
			self.mergeHistory += [[clusMem[group1], clusMem[group2]]]
		

if __name__=='__main__':
	N, Nclass, Nquery = (300, 6, 20)
	data, labels = ds.make_blobs(n_samples=N, n_features=2, centers=Nclass)
	a = ActiveClassDiscovery(data)
	trainInd = np.zeros(Nquery).astype(int)
	for i in range(Nquery):
		trainInd[i] = a.get_query()
		plt.figure()
		cc.plot_labels(data)
		cc.plot_labels(data[trainInd[:i+1]], labels[trainInd[:i+1]])
		plt.show()
					
	

