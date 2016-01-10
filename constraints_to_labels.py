import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as ds
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors

import constrained_clustering as cc


class ConstraintsToLabels(cc.ConstrainedClustering):
	"""Using the unsupervised structure of the data provided 
	by a hierarchical clustering of the data, along with the 
	information provided by pairwise constraints, turn a set
	of constraints into a set of labeled samples
	
	Attributes
	----------
	n_clusters : desired number of clusters
	hierarchical : AgglomerativeClustering object from sklearn
		       this contains the merges that occur during
		       an unsupervised run of hierarchical clustering	
		       of the data
	labelSet : array containing the labels for all the samples		      
		   contained in the array 'constrainedSamples'
		   from the parent class
	"""
	def __init__(self, **kwargs): 
		super( ConstraintsToLabels, self).__init__(**kwargs)
		self.hierarchical = AgglomerativeClustering(linkage='average')
	
	def fit_constrained(self):
		"""Transform a set of pairwise constraints into a set of
		labeled training data
		"""
		# Unsupervised hierarchical clustering
		self.hierarchical.fit(self.data)
		self.N,_ = self.data.shape
		
		self.labelSet = np.arange(0,self.constrainedSamps.size)
		# Find groups of constrained samples such that no CL 
		# constraints are found within a single group
		self.agglomerate_constrained_samples()
		
		# Merge the nodes representing agglomerated clusters of 
		# constrained samples to best obey ML and CL constraints 
		newLabels = self.merge_nodes()
		self.translate_labels(newLabels)

	def merge_nodes(self):
		"""We have an oversegmentation of the constrained points we wish
		to assign labels to, represented by the attribute self.labelSet.
		This segmentation obeys CL constraints. Call each group of samples
		with the same labels a node. We find the net constraint value 
		between nodes, and group nodes to produce a proper label set for 
		the constrained samples.
		"""
		uniqueLabels = np.unique(self.labelSet)
		NuniqueLabels = uniqueLabels.size
		
		groupCenters = np.zeros((NuniqueLabels,self.data.shape[1]))
		groupPop = np.zeros((NuniqueLabels,1))
		# simMat contains the Nml - Ncl net constraint value
		self.simMat = np.zeros((NuniqueLabels,NuniqueLabels))
		# Loop over the node labels
		for i in range(NuniqueLabels):
			group1 = self.constrainedSamps[self.labelSet==i]
			groupCenters[i,:] = np.mean(self.data[group1,:], axis=0)
			groupPop[i] = group1.size
			if i < (NuniqueLabels-1):
				# Loop over all other nodes
				for ii in range(i+1,NuniqueLabels):
					group2 = self.constrainedSamps[self.labelSet==ii]
					Ncl = self.number_of_constraints(group1,group2,0)
					Nml = self.number_of_constraints(group1,group2,1)
					val = Nml - Ncl
					self.simMat[i,ii] = val
					self.simMat[ii,i] = val
		#Nneigh = 2
		#if groupCenters.shape[0] > (Nneigh*2):
		#	self.complete_matrix(groupCenters, 2)
		self.plot_graph_cut_problem(groupCenters, groupPop)
		
		centerDists = cc.squared_distance_matrix(groupCenters)
		return self.graph_cut_approx(centerDists)

	def graph_cut_approx(self, centerDists):
		"""Find which of the nodes will merge based on the +/- constraint
		values between nodes. This is a very simple implementation, and will
		not produce a proper graph cut solution. Basically, we just merge
		two nodes if there is a net positive ML between them.
		"""
		Nnodes = self.simMat.shape[0]
		newLabels = np.arange(Nnodes)
		Ngroups = Nnodes
		
		self.simMat += np.diag(-np.inf*np.ones(Nnodes))		

		keepLooping = True
		while keepLooping:
			maxInd = np.argmax(self.simMat) 
			r, c = np.unravel_index(maxInd, self.simMat.shape)
			maxVal = self.simMat[r,c]			
			if maxVal > 0:
				newLabels[newLabels==newLabels[c]] = newLabels[r]
				self.merge_columns(r, c)
				Ngroups -= 1
			elif self.n_clusters is not None:
				if maxVal==-np.inf:
					keepLooping = False
				elif Ngroups > self.n_clusters:
					if maxVal==0:
						r,c = self.break_tie_zero(centerDists)			
					newLabels[newLabels==newLabels[c]] = newLabels[r]
					self.merge_columns(r, c)	
					Ngroups -= 1	
				else:
					keepLooping = False
			else:
				keepLooping = False
		return newLabels
	
	def break_tie_zero(self, centerDists):
		"""This is just an approx because when two centers merge
		I am not adjusting the centerDists. If it doesn't work we
		can fix it
		"""
		r,c = np.triu_indices(self.simMat.shape[0], 1)
		zeroInd = self.simMat[r,c]==0	
		rowZero = r[zeroInd]
		colZero = c[zeroInd]
		zeroDists = centerDists[rowZero, colZero]	
		bestPair = np.argmin(zeroDists)	
		return rowZero[bestPair], colZero[bestPair]	
			
	def merge_columns(self, r, c):
		self.simMat[c,:] += self.simMat[r,:]
		self.simMat[:,c] = self.simMat[c,:]
		self.simMat[r,:] = -np.inf
		self.simMat[:,r] = -np.inf
		self.simMat[c,c] = -np.inf
		
	def plot_graph_cut_problem(self, centers, nodeName):
		"""By using the agglomerative property of hierarchical 
		clustering, samples involved in pairwise constraints
		are grouped into an oversegmentation of the data. These
		segments can be represented by nodes, with a net +/- sum 
		of ML and CL constraints between them. Thus, we have a 
		graph cut problem. 

		This function plots the samples involved in constraints,
		plots the nodes representing the agglomerated groups, marks
		the number of samples associated with each node, and plots
		lines representing the net constraint value between nodes 
		with thickness proportional to the number of constraints.

		Parameters
		----------
		centers - matrix containing node locations
		nodeName - list containing the value that will be placed
			   in each node 
		"""
		plt.figure()
		uniquePairs = np.triu(self.simMat, k=1)
		row,col = np.nonzero(uniquePairs)		
		
		# Plot the constrained samples
		cc.plot_labels(self.data[self.constrainedSamps,:],self.labelSet)
		
		# Plot lines between nodes representing the number and type
		# of constraints between them
		maxSim = np.max(np.abs(self.simMat[row,col]))
		for r,c in zip(row,col):
			if self.simMat[r,c] > 0:
				lineType = '-'
				lineColor='b'
			else:	
				lineType = '--'
				lineColor='r'
			lineThick = np.abs(self.simMat[r,c])
			lineThick *= 10/maxSim
			
			plt.plot(centers[[r,c],0],centers[[r,c],1],lineType,
			         color=lineColor,
				 linewidth=lineThick)

		# Plot the nodes themselves
		plt.plot(centers[:,0], centers[:,1], 'o', 
		         markersize=20,
			 markerfacecolor=[0.7,0.7,0.7])
		
		# Put the population of the node in the center
		for i, val in enumerate(nodeName):
			plt.text(centers[i,0], centers[i,1], str(val),
				 horizontalalignment='center',
				 verticalalignment='center') 
		plt.show()
	
	def complete_matrix(self, centers, k):
		"""For the 'floating nodes' that are not connected to anything
		else, add a weak ML connection to their nearest neighbors.
		
		Parameters
		----------
		centers - location of the centers of the nodes in the data space
		k - number of nearest neighbors to look for
		"""
		assert k >= 1
		Nnodes = self.simMat.shape[0]
		nbrs = NearestNeighbors(n_neighbors=k+1, 
					algorithm='ball_tree').fit(centers)
		distances, indices = nbrs.kneighbors(centers)
		oldSimMat = self.simMat.copy()
		for i in range(self.simMat.shape[0]):
			neighInd = np.zeros((1,Nnodes)) > 0
			neighInd[:,indices[i,1:(k+1)]] = True
			neighInd = (neighInd & (oldSimMat[i,:]==0)).reshape((-1,))
			self.simMat[i,neighInd] = 1
			self.simMat[neighInd,i] = 1			

	def translate_labels(self, newLabels):
		"""We have clustered nodes, so we need to reflect these changes
		in the labelSet, which contains our generated labels for the
		constrained samples.
		""" 
		uniqueLabels = np.unique(self.labelSet)
		for lab in uniqueLabels:
			self.labelSet[self.labelSet==lab] = newLabels[lab]

	def agglomerate_constrained_samples(self):
		"""Given the merges of the hierarchical clustering, iterate 
		through them from the lowest level to the highest level. Make
		the labels of constrained samples the same if they are present
		in the same merge group with no CL constraints violated.
		"""
		bigLabelSet = -np.ones((self.N,1))
		bigLabelSet = np.arange(self.N)
		#bigLabelSet[self.constrainedSamps] = self.labelSet.reshape((-1,1))

		allMerges = self.linkage_to_merges()	
		for merge in allMerges:	
			group1 = merge[0]
			group2 = merge[1]
			
			allSamps = np.append(group1, group2)	
			if not self.is_CL_violated(allSamps):	
				bigLabelSet[group1] = bigLabelSet[group2[0]]
				
		newLabels = bigLabelSet[self.constrainedSamps]
		newLabels = cc.translate_to_counting_numbers(newLabels)
		self.labelSet = newLabels	

	def linkage_to_merges(self):
		"""Hierarchical clustering returns a matrix of 
		merges, starting with samples 0 to N-1, then 
		adding new labels for groups of merged samples.
		This is a generator function that returns 
		a list of the two arrays containing the indices 
		of all the samples in the two merged groups
		"""
		self.mergeInd = []
		clusMem = [np.asarray([x]) for x in range(self.N)]
		for i in range(0,self.N-1):
			group1 = self.hierarchical.children_[i,0]
			group2 = self.hierarchical.children_[i,1]
			clusMem += [np.append( clusMem[group1], clusMem[group2])]
			yield [clusMem[group1], clusMem[group2]]		

if __name__ == '__main__':
	# Parameters---------------------------------------
	Nclusters = 4
	N = 1000
	Nconstraints = 100
	#---------------------------------------------------
	# Make some synthetic data
	data, labels = ds.make_blobs(n_samples=N, 
				     n_features=2, 
                                     centers=Nclusters)
	
	# Make some constaints	
	constraintMat = cc.ConstrainedClustering.make_constraints(labels,
						Nconstraints=Nconstraints,
						errRate=0.0)

	# Plot the data along with the constraints
	plt.figure()
	cc.ConstrainedClustering.plot_constraints(data, constraintMat)
	plt.show()
	
	# Turn the pairwise constraints into labeled samples
	ctlObj = ConstraintsToLabels(data=data, 
				     constraintMat=constraintMat, 
                                     n_clusters=Nclusters)
	ctlObj.fit_constrained()

	# Now these labels and their associated index can be used
	# in a classifier instead of clustering
	trainLabels = ctlObj.labelSet
	trainInd = ctlObj.constrainedSamps

	# Plot the resulting data, along with the training samples
	# that were produced from the pairwise constraints
	plt.figure()
	cc.plot_labels(data)
	cc.plot_labels(data[trainInd,:],trainLabels)	
	plt.show()
