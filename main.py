import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as ds
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors


def distance_matrix(dataMatrix):
	"""From the n x d data matrix, compute the n x n
	distance matrix
	"""
	dataTrans = np.transpose(dataMatrix[:,:,None], (2,1,0))
	diff = dataMatrix[:,:,None] - dataTrans
	return np.sum( diff**2 )


def ismember(a,b):
	"""Return an array with the same size of 'a' that 
	indicates if an element of 'a' is in 'b'. Can be 
	used as an index for slicing since it contains bool 
	elements
	"""
	a = np.asarray(a)
	b = np.asarray(b)
	memberInd = np.zeros_like(a)
	for element in b:
		memberInd[a==element] = 1
	return memberInd>0


def translate_to_counting_numbers(a):
	"""Perform a 1-to-1 mapping of the numbers from a numerical
	array to a new array with elements that are in the set
	{0,1,...,M-1} where M is the number of unique elements in 
	the original array 
	"""
	a = np.asarray(a)
	uniqueElements = np.unique(a)
	
	boolBlock = a.reshape((-1,1)) == uniqueElements.reshape((1,-1))
 	newValueInMatrix = boolBlock.astype(int) * np.arange(uniqueElements.size)	
	return np.sum(newValueInMatrix,axis=1)


def plot_labels(data,labels):
	"""Plot the data colored according to the unique class 
	labels
	"""
	classes = set(labels)
	colors = plt.cm.Spectral(np.linspace(0,1,len(classes)))

	for lab, col in zip(classes, colors):
		ind = labels == lab
		plt.plot(data[ind,0], data[ind,1], 'o', 
			 markerfacecolor=col, 
			 markersize=10)


class ConstrainedClustering(object):
	"""A class useful as a parent for  constrained clustering 
	algorithms
	
	Attributes
	----------
	data : the n x d data matrix
	constraintMat : the m x 3 constraint matrix, where m is 
			the total number of constraints. Each row 
			contains the indices of the two samples	
			involved in the constraint and a value 0
			or 1 for CL or ML
	constrainedSamps : array containing the indices of samples 
         		   involved in a constraint
	ML : each row contains must-link index pairs
	CL : each row contains cannot-link index pairs
	 
	Methods
	-------
	constraints_by_value(constraintMat,consVal) 
	    - Return _ x 2 matrix containing index pairs for 
	      constraints with value of consVal
	is_CL_violated(group)
	    - Return True if a CL constraint is present within 
	      the samples in 'group'
	number_of_constraints(group1,group2,consVal)
	    - Return the number of constraints of value 'consVal' 
	      between 'group1' and 'group2'
	plot_constraints()
	    - Plot the pairwise constraints and the data
	make_constraints(labels)
	    - Given the true set of labels for the dataset, 
	      produce a set of synthetically generated constraints
	"""

	def __init__(self,data,constraintMat):
		self.data = data

		ML = self.constraints_by_value(constraintMat,1)
		self.ML = np.append(ML,ML[:,-1::-1],axis=0)
		CL = self.constraints_by_value(constraintMat,0)
		self.CL = np.append(CL,CL[:,-1::-1],axis=0)
		
		self.constrainedSamps = np.unique( constraintMat.reshape(-1,1) )

	def constraints_by_value(self,constraintMat,consVal):
		ind = constraintMat[:,2]==consVal
		return constraintMat[ind,0:2]
	
	def transitive_closure(self):
		pass	
		
	def other_sample_in_pair(self, group, consVal):
		assert consVal==0 or consVal==1
		if consVal == 0:
			constraintBlock = self.CL
		elif consVal == 1:
			constraintBlock = self.ML

		involvedInConstraint = ismember(constraintBlock[:,0], group)
		return constraintBlock[involvedInConstraint,1]
		
	def is_CL_violated(self, group):
		otherCLsamp = self.other_sample_in_pair(group,0)
		isAlsoInGroup = ismember(group,otherCLsamp)
		return np.any(isAlsoInGroup)

	def number_of_constraints(self, group1, group2, consVal):
		otherSamp1 = self.other_sample_in_pair(group1,consVal)
		isInGroup2 = ismember(group2,otherSamp1)
		return np.sum(isInGroup2)

	def plot_constraints(self):
		"""Plot the data (all grey) and the pairwise 
		constraints
		ML constraints will be solid lines, while CL 
		constraints will be dashed lines
		"""
		f = plt.figure()
		for ml in self.ML:
			plt.plot(data[ml,0], data[ml,1], '-', color='black', linewidth=5)
		for cl in self.CL:
			plt.plot(data[cl,0],data[cl,1],'--',color='black',linewidth=5)
		aNiceGrey = np.ones(4)
		aNiceGrey[0:3] *= 0.3
		plt.plot(data[:,0],data[:,1],'o',markerfacecolor=aNiceGrey,markersize=10)
		f.show()
			
	@staticmethod 
	def make_constraints( labels, Nconstraints=None, errRate=0):
		N = len(labels)
		# Make random constraints, good for testing
		if Nconstraints is None:
			# Half the number of samples is a good baseline
			Nconstraints = len(labels)/2

		# Just the pairs of indices involved in each constraint
		queryMat = np.random.randint(0,N,(Nconstraints,2))
		link = (labels[queryMat[:,0]] == labels[queryMat[:,1]])+0
		# The samples whose link values we will invert
		errorInd = np.random.choice(2,Nconstraints,p=[1-errRate,errRate])		
		link = link.reshape((-1,1))
		link[errorInd,:] = 2 - np.power(2,link[errorInd,:])

		constraintMat = np.append(queryMat,link,axis=1)
		return constraintMat


class ConstraintsToLabels(ConstrainedClustering):
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

	def __init__(self, n_clusters=None, **kwargs): 
		super( ConstraintsToLabels, self).__init__(**kwargs)
		self.n_clusters = n_clusters 
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
		self.merge_labels()

		
		similarityMatrix = self.compute_similarities()
		specClus = SpectralClustering(n_clusters=self.n_clusters,affinity='precomputed')
		specClus.fit(similarityMatrix)
		self.translate_labels(specClus.labels_)

	def translate_labels(self,newLabels):
		uniqueLabels = np.unique(self.labelSet)
		for lab in uniqueLabels:
			self.labelSet[self.labelSet==lab] = newLabels[lab]

	def compute_similarities(self):
		uniqueLabels = np.unique(self.labelSet)
		NuniqueLabels = uniqueLabels.size
		
		groupCenters = np.zeros((NuniqueLabels,self.data.shape[1]))
		groupPop = np.zeros((NuniqueLabels,1))
		plusMinusMat = np.zeros((NuniqueLabels,NuniqueLabels))
		for i in range(NuniqueLabels):
			group1 = self.constrainedSamps[self.labelSet==i]
			groupCenters[i,:] = np.mean(self.data[group1,:], axis=0)
			groupPop[i] = group1.size
			if i < (NuniqueLabels-1):
				for ii in range(i+1,NuniqueLabels):
					group2 = self.constrainedSamps[self.labelSet==ii]
					Ncl = self.number_of_constraints(group1,group2,0)
					Nml = self.number_of_constraints(group1,group2,1)
					val = Nml - Ncl
					plusMinusMat[i,ii] = val
					plusMinusMat[ii,i] = val
		
		plusMinusMat = self.complete_matrix(groupCenters, plusMinusMat, 2)
		self.plot_graph_cut_problem(groupCenters, groupPop, plusMinusMat)
		simMat = 0.5*(np.tanh(plusMinusMat)+1)
		simMat += 0.5*np.identity(NuniqueLabels)	
		return simMat

	def merge_viable(self ): 
		pass

	def plot_graph_cut_problem(self, centers, nodeName, simMat):
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
		simMat - similarity matrix between nodes (Nml - Ncl) 
		"""
		plt.figure()
		uniquePairs = np.triu(simMat, k=1)
		row,col = np.nonzero(uniquePairs)		
		
		# Plot the constrained samples
		plot_labels(self.data[self.constrainedSamps,:],self.labelSet)	
		
		# Plot lines between nodes representing the number and type
		# of constraints between them
		maxSim = np.max(np.abs(simMat[row,col]))
		for r,c in zip(row,col):
			if simMat[r,c] > 0:
				lineType = '-'
				lineColor='b'
			else:	
				lineType = '--'
				lineColor='r'
			lineThick = np.abs(simMat[r,c])
			lineThick *= 10/maxSim
			
			print np.max(lineThick)

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
	
	def complete_matrix(self, centers, simMat, k):
		"""For the 'floating nodes' that are not connected to anything
		else, add a weak ML connection to their nearest neighbors.
		
		Parameters
		----------
		centers - location of the centers of the nodes in the data space
		simMat - similarity matrix represented by the +/- of the ML and
		         CL total between nodes
		k - number of nearest neighbors to look for
			
		Returns
		-------
		newSimMat - similarity matrix with some ML connections added
		"""
		assert k >= 1

		nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(centers)
		distances, indices = nbrs.kneighbors(centers)
		newSimMat = simMat.copy()
		for i in range(simMat.shape[0]):
			if np.all(simMat[i,:]==0):
				neighInd = indices[i,i:(k+1)]
				newSimMat[i,neighInd] = 1
				newSimMat[neighInd,i] = 1
		return newSimMat

	def merge_labels(self):
		"""Given the merges of the hierarchical clustering, iterate 
		through them from the lowest level to the highest level. Make
		the labels of constrained samples the same if they are present
		in the same merge group with no CL constraints violated.
		"""
		bigLabelSet = -np.ones((self.N,1))
		bigLabelSet[self.constrainedSamps] = self.labelSet.reshape((-1,1))

		allMerges = self.linkage_to_merges()	
				
		for merge in allMerges:	
			group1 = merge[0]
			group2 = merge[1]
			
			allSamps = np.append(group1, group2)		
			if not self.is_CL_violated(allSamps):	
				bigLabelSet[group1] = bigLabelSet[group2[0]]
		
		newLabels = bigLabelSet[self.constrainedSamps]
		newLabels = translate_to_counting_numbers(newLabels)
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

	# Make some synthetic data
	Nclusters = 3
	N = 100
	data,labels = ds.make_blobs(n_samples=N, n_features=2, centers=Nclusters)

	np.set_printoptions(precision=2,suppress=True)	
	
	# Make some constaints	
	constraintMat = ConstrainedClustering.make_constraints(labels,Nconstraints=100)
	ctlObj = ConstraintsToLabels(data=data, constraintMat=constraintMat, n_clusters=Nclusters)
	ctlObj.fit_constrained()
	trainLabels = ctlObj.labelSet
	trainInd = ctlObj.constrainedSamps

	#ctlObj.plot_constraints()
	#plot_labels(data[trainInd,:],trainLabels)	
