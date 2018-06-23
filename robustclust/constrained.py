import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as ds
from sklearn.cluster import AgglomerativeClustering, \
                            SpectralClustering, \
                            KMeans
from sklearn.neighbors import KNeighborsClassifier, \
                              NearestNeighbors


class ConstrainedClustering:
  """A class useful as a parent for  constrained clustering 
  algorithms
  
  Attributes
  ----------
  data : the n x d data matrix
  constraint_mat : the m x 3 constraint matrix, where m is 
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
  constraints_by_value(constraint_mat,consVal) 
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
  def __init__(self, data, constraint_mat, n_clusters=None):
    self.data = data
    self.n_clusters = n_clusters

    ML = self.constraints_by_value(constraint_mat,1)
    self.ML = np.append(ML,ML[:,-1::-1],axis=0)
    CL = self.constraints_by_value(constraint_mat,0)
    self.CL = np.append(CL,CL[:,-1::-1],axis=0)
    
    self.constrainedSamps = np.unique(constraint_mat.reshape(-1,1))

  def constraints_by_value(self,constraint_mat,consVal):
    ind = constraint_mat[:,2]==consVal
    return constraint_mat[ind,0:2]
  
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


class E2CP(ConstrainedClustering):
  """Exhaustive and efficient constraint propagation by Lu
  """
  def __init__(self, k_E2CP=15, alpha=0.6, **kwargs):
    super(E2CP, self).__init__(**kwargs)
    assert self.n_clusters is not None
    self.k_E2CP = np.min([self.data.shape[0]-1, k_E2CP])
    self.alpha = alpha

  def fit_constrained(self):
    N = self.data.shape[0]
    self.affMat = get_affinity(data=self.data)
    nbrs = NearestNeighbors(n_neighbors=self.k_E2CP+1,
         algorithm='ball_tree').fit(self.data)
    distances, indices = nbrs.kneighbors(self.data)
    W = np.zeros(self.affMat.shape)
    
    ind1 = (np.arange(N).reshape((-1,1)) * np.ones((1,self.k_E2CP))).reshape(-1).astype('int')
    ind2 = indices[:,1:].reshape(-1).astype('int')
    W[ind1, ind2] = self.affMat[ind1, ind2] / (np.sqrt(self.affMat[ind1, ind1]) * np.sqrt(self.affMat[ind2, ind2]))
    
    W = (W+W.transpose())/2
    Dsqrt = np.diag( np.sum(W, axis=1)**-0.5 )  
    Lbar = np.dot(np.dot(Dsqrt, W), Dsqrt)
    
    Z = np.zeros(self.affMat.shape)
    Z[self.ML[:,0], self.ML[:,1]] = 1
    Z[self.CL[:,0], self.CL[:,1]] = -1
  
    Fv = np.zeros(Z.shape)
    for i in range(50):
      Fv = self.alpha*np.dot(Lbar, Fv) + (1-self.alpha)*Z
  
    Fh = np.zeros(Z.shape)
    for i in range(50):
      Fh = self.alpha*np.dot(Fh, Lbar) + (1-self.alpha)*Fv

    Fbar = Fh / np.max( np.abs( Fh.reshape(-1) ) )
    
    Wbar = np.zeros(self.affMat.shape)
    mlInd = Fbar >= 0
    Wbar[mlInd] = 1 - (1 - Fbar[mlInd]) * (1 - W[mlInd])
    clInd = Fbar < 0
    Wbar[clInd] = (1 + Fbar[clInd]) * W[clInd]    
    
    specClus = SpectralClustering(n_clusters=self.n_clusters,
                affinity='precomputed')
    specClus.fit(Wbar)
    self.labels = specClus.labels_


class SpectralLearning(ConstrainedClustering):
  """Spectral Learning by Kamvar
  """  
  def __init__(self, **kwargs):
    super(SpectralLearning, self).__init__(**kwargs)
    assert self.n_clusters is not None    

  def fit_constrained(self):
    self.affMat = get_affinity(data=self.data)  
    self.apply_constraints()  
    newData = self.laplacian_eig()
    
    kmeans = KMeans(n_clusters=self.n_clusters)  
    kmeans.fit(newData)
    self.labels = kmeans.labels_    
      
  def laplacian_eig(self):
    rowSums = np.sum(self.affMat, axis=1)
    dmax = np.max(rowSums)
    D = np.diag(rowSums)
    L = (self.affMat + dmax*np.eye(D.shape[0]) - D)/dmax
    
    values, vectors = np.linalg.eig(L)  
    assert np.all( np.isreal(values) )
    
    bigEigInd = np.argsort(-values)
    return vectors[:,bigEigInd[:self.n_clusters]]
      
  def apply_constraints(self):
    self.affMat[self.ML[:,0], self.ML[:,1]] = 1
    self.affMat[self.CL[:,0], self.CL[:,1]] = 0  


if __name__=='__main__':
  Nclusters, N, Nconstraints = (3, 100, 40)
  data,labels = ds.make_blobs(n_samples=N, n_features=2, centers=Nclusters)
  
  constraint_mat = ConstrainedClustering.make_constraints(labels, data=data, method='mmffqs', Nconstraints=Nconstraints, errRate=0)
  
  plt.figure()
  ConstrainedClustering.plot_constraints(data, constraint_mat)
  plt.show()
