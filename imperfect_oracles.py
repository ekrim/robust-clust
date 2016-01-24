import numpy as np
from sklearn.cluster import KMeans, SpectralClustering, MeanShift, AffinityPropagation, AgglomerativeClustering, DBSCAN
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
import sklearn.datasets as ds
import matplotlib.pyplot as plt
import constrained_clustering as cc


class ImperfectOracles(cc.ConstrainedClustering):
	"""Given a set of constraints and a dataset, determine
	which ones are harmful to the clustering process. This
	is determined by analyzing the disagreement within the 
	constraint set. We build an ensemble of unsupervised 
	clusterings. If two constraints can be mutually satisfied
	at some point in that ensemble, then they vote for each 
	other. Only constraints in the same local regions can vote
	for one another. A measure of 'trust' is computed based
	on votes, and it is calculated iteratively to reveal the 
	constraints which should be kept. 
	"""
	def __init__(self, threshold=0.8, **kwargs):
		super(ImperfectOracles, self).__init__(**kwargs)
		self.threshold = threshold
		self.constraintMat = kwargs['constraintMat'] 
	
	def remove_constraints(self):	
		Ncons = self.constraintMat.shape[0]
		self.voteEligible = np.zeros((Ncons,Ncons))
		
		voteMat = np.zeros((Ncons,Ncons)) > 0
		satisfiedConstraints = np.zeros(Ncons) > 0
			
		CL = self.CL[:self.CL.shape[0]/2,:]
		ML = self.ML[:self.ML.shape[0]/2,:]
		for i in range(4):
			representation = self.data
			if i==0: 
				clus = KMeans(n_clusters=self.n_clusters)
			elif i==1:
				representation = cc.get_affinity(data=self.data)
				clus = SpectralClustering(n_clusters=self.n_clusters, affinity='precomputed')	
			elif i==2:
				clus = MeanShift()
			elif i==3:
				clus = AgglomerativeClustering(n_clusters=self.n_clusters, linkage='average')
			"""elif i==4:
				distMat = np.sqrt(cc.squared_distance_matrix(self.data))
				meanNN = np.mean(np.sort(distMat, axis=1)[:,1])
				clus = DBSCAN(eps=4*meanNN, min_samples=5 )
			"""
			labs = clus.fit_predict(representation)
			
			"""plt.figure()
			cc.plot_labels(self.data, labels=labs)
			plt.show()
			"""
			self.determine_relevance(labs)

			connMat = labs[:,None]==labs[None,:]
			tempSat = connMat[self.constraintMat[:,0],self.constraintMat[:,1]]==self.constraintMat[:,2]	
			satisfiedConstraints = satisfiedConstraints | tempSat
			voteMat = voteMat | (tempSat[:,None]==tempSat[None,:])	

		# Now we have determined which constraints are relvant
		# and which ones vote for one another
		# Mask the unsatisfied constraints, and the ones of same
		# type (ML or CL) from voting for each other
		self.voteEligible = self.voteEligible*satisfiedConstraints[None,:]
		neqMask = np.logical_xor(self.constraintMat[:,2][:,None], self.constraintMat[:,2][None,:]).astype(int)
		
		self.voteEligible *= neqMask
		voteMat *= self.voteEligible
		
		trustedNess = 0.5*np.ones(Ncons)
		trustedNess[np.logical_not(satisfiedConstraints)] = 0
		trustMat = np.zeros((Ncons,11))
		trustMat[:,0] = trustedNess
		for i in range(1,11): 
			possibleTrustVotes = np.dot(self.voteEligible, trustedNess)
			receivedTrustVotes = np.dot(voteMat, trustedNess).astype(float)
			trustedNess = receivedTrustVotes/possibleTrustVotes
			
			trustedNess[possibleTrustVotes==0] = 1
			trustMat[:,i] = trustedNess
	
		keepInd = trustedNess >= self.threshold
		return keepInd

	def plot_removal(self, trueLabels, keepInd):
		errInd = self.find_errors(trueLabels, keepInd)

		# The ones we kept
		plt.subplot(2,2,1)
		cc.plot_labels(self.data, labels=trueLabels)
		plt.title('True grouping')		

		# Errors
		plt.subplot(2,2,2)
		cc.ConstrainedClustering.plot_constraints(self.data, self.constraintMat[errInd,:])
		plt.title('Oracle errors')
		
		# Good ones we removed
		plt.subplot(2,2,3)
		tossInd = np.logical_not(keepInd)
		correctInd = np.logical_not(errInd)
		correctTossed = np.logical_and(tossInd, correctInd)
		cc.ConstrainedClustering.plot_constraints(self.data, self.constraintMat[correctTossed,:])
		plt.title('Correct constraints removed')		

		# Errors we left
		plt.subplot(2,2,4)
		cc.ConstrainedClustering.plot_constraints(self.data, self.constraintMat[np.logical_and(keepInd, errInd),:])	
		plt.title('Errors remaining')

	def find_errors(self, trueLabels, keepInd):
		trueConnMat = (trueLabels[:,None]==trueLabels[None,:]).astype(int)
		trueCons = trueConnMat[self.constraintMat[:,0], self.constraintMat[:,1]]
		errInd = trueCons != self.constraintMat[:,2]
		return errInd
			
	def determine_relevance(self, labs):
		for i, cons in enumerate(self.constraintMat):
			clus1 = labs[cons[0]]
			clus2 = labs[cons[1]]	
			col1 = labs[self.constraintMat[:,0]][:,None]
			col2 = labs[self.constraintMat[:,1]][:,None]
			labsOfCons = np.concatenate((col1,col2), axis=1)
			Nmatches1 = np.sum(labsOfCons==np.asarray([clus1,clus2])[None,:], axis=1)
			Nmatches2 = np.sum(labsOfCons==np.asarray([clus2,clus1])[None,:], axis=1)
			Nmatches = np.maximum(Nmatches1, Nmatches2)
			if cons[2]==1:
				validInd = Nmatches==2
			else:
				validInd = Nmatches>=1
			self.voteEligible[i,validInd] = 1		


if __name__=='__main__':
	N, Nclusters, Nconstraints, errRate = (200,3,50,0.1)
	data, labels = ds.make_blobs(n_samples=N, 
				     n_features=2, 
				     centers=Nclusters)

	constraintMat = cc.ConstrainedClustering.make_constraints(labels,
						Nconstraints=Nconstraints,
						errRate=errRate)	
	a = ImperfectOracles(data=data,
			     constraintMat=constraintMat,
			     n_clusters=Nclusters)
	keepInd = a.remove_constraints()

	plt.figure()
	a.plot_removal(labels, keepInd)
	plt.tight_layout()
	plt.show()

