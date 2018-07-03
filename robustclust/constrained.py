import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, \
                                   squareform
import sklearn.datasets as ds
from sklearn.cluster import AgglomerativeClustering, \
                            SpectralClustering, \
                            KMeans
from sklearn.neighbors import KNeighborsClassifier, \
                              NearestNeighbors

from .utils import affinity, \
                   ismember
from .active_learning import get_constraints


class ConstrainedClustering:
  """Base class for constrained clusterers

  Methods
  -------
  constraints_by_value(constraint_mat,cons_val) 
      - Return _ x 2 matrix containing index pairs for 
        constraints with value of cons_val
  is_CL_violated(group)
      - Return True if a CL constraint is present within 
        the samples in 'group'
  number_of_constraints(group1,group2,cons_val)
      - Return the number of constraints of value 'cons_val' 
        between 'group1' and 'group2'
  plot_constraints()
      - Plot the pairwise constraints and the data
  make_constraints(labels)
      - Given the true set of labels for the dataset, 
        produce a set of synthetically generated constraints

  """
  def __init__(self, constraint_mat=None, n_clusters=None):
    """
    Args:
      data: the (n,d) data matrix (ndarray)
      constraint_mat: the m x 3 constraint matrix, where m is 
        the total number of constraints. Each row 
        contains the indices of the two samples  
        involved in the constraint and a value 0
        or 1 for CL or ML
      constrained_samps: array containing the indices of samples 
        involved in a constraint
      ML: each row contains must-link index pairs
      CL: each row contains cannot-link index pairs
  
    """
    self.n_clusters = n_clusters

    ML = self.constraints_by_value(constraint_mat, 1)
    self.ML = np.append(ML, ML[:,-1::-1], axis=0)
    CL = self.constraints_by_value(constraint_mat, 0)
    self.CL = np.append(CL, CL[:,-1::-1], axis=0)
    
    self.constrained_samps = np.unique(constraint_mat.reshape(-1,1))

  def constraints_by_value(self, constraint_mat, cons_val):
    ind = constraint_mat[:,2] == cons_val
    return constraint_mat[ind, 0:2]
  
  def transitive_closure(self):
    pass  
    
  def other_sample_in_pair(self, group, cons_val):
    assert cons_val in [0,1], 'invalid constraint value'
    if cons_val == 0:
      constraint_block = self.CL
    elif cons_val == 1:
      constraint_block = self.ML

    involved_in_constraint = ismember(constraint_block[:,0], group)
    return constraint_block[involved_in_constraint, 1]
    
  def is_CL_violated(self, group):
    other_CL_samp = self.other_sample_in_pair(group, 0)
    is_also_in_group = ismember(group, other_CL_samp)
    return np.any(is_also_in_group)

  def number_of_constraints(self, group1, group2, cons_val):
    other_samp1 = self.other_sample_in_pair(group1, cons_val)
    is_in_group2 = ismember(group2, other_samp1)
    return np.sum(is_in_group2)


class E2CP(ConstrainedClustering):
  """Exhaustive and efficient constraint propagation by Lu

  """
  def __init__(self, k_E2CP=15, alpha=0.6, **kwargs):
    super().__init__(**kwargs)
    assert self.n_clusters is not None
    self.k_E2CP = k_E2CP
    self.alpha = alpha

  def fit_constrained(self, data):
    N = data.shape[0]
    self.k_E2CP = np.min([N-1, self.k_E2CP])
    self.aff_mat = squareform(affinity(pdist(data)))

    nbrs = NearestNeighbors(
      n_neighbors=self.k_E2CP+1,
      algorithm='ball_tree').fit(data)

    distances, indices = nbrs.kneighbors(data)
    W = np.zeros(self.aff_mat.shape)
    
    ind1 = (np.arange(N)[:,None] * np.ones((1, self.k_E2CP))).flatten().astype('int')
    ind2 = indices[:, 1:].flatten().astype('int')
    W[ind1, ind2] = self.aff_mat[ind1, ind2] / (np.sqrt(self.aff_mat[ind1, ind1]) * np.sqrt(self.aff_mat[ind2, ind2]))
    
    W = (W + W.transpose())/2
    Dsqrt = np.diag( np.sum(W, axis=1)**-0.5 )  
    Lbar = np.dot(np.dot(Dsqrt, W), Dsqrt)
    
    Z = np.zeros(self.aff_mat.shape)
    Z[self.ML[:,0], self.ML[:,1]] = 1
    Z[self.CL[:,0], self.CL[:,1]] = -1
  
    Fv = np.zeros(Z.shape)
    for i in range(50):
      Fv = self.alpha*np.dot(Lbar, Fv) + (1-self.alpha)*Z
  
    Fh = np.zeros(Z.shape)
    for i in range(50):
      Fh = self.alpha*np.dot(Fh, Lbar) + (1-self.alpha)*Fv

    Fbar = Fh / np.max( np.abs( Fh.reshape(-1) ) )
    
    Wbar = np.zeros(self.aff_mat.shape)
    ml_ind = Fbar >= 0
    Wbar[ml_ind] = 1 - (1 - Fbar[ml_ind]) * (1 - W[ml_ind])
    cl_ind = Fbar < 0
    Wbar[cl_ind] = (1 + Fbar[cl_ind]) * W[cl_ind]    
    
    specClus = SpectralClustering(
      n_clusters=self.n_clusters,
      affinity='precomputed')
    specClus.fit(Wbar)
    self.labels = specClus.labels_


class SpectralLearning(ConstrainedClustering):
  """Spectral Learning by Kamvar

  """  
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    assert self.n_clusters is not None, \
      'must specify number of clusters'

  def fit_constrained(self, data):
    self.aff_mat = squareform(affinity(pdist(data)))
    self._apply_constraints()  
    new_data = self._laplacian_eig()
    
    kmeans = KMeans(n_clusters=self.n_clusters)  
    kmeans.fit(new_data)
    self.labels = kmeans.labels_    
      
  def _laplacian_eig(self):
    row_sums = np.sum(self.aff_mat, axis=1)
    dmax = np.max(row_sums)
    D = np.diag(row_sums)
    L = (self.aff_mat + dmax*np.eye(D.shape[0]) - D)/dmax
    
    values, vectors = np.linalg.eig(L)  
    assert np.all( np.isreal(values) )
    
    big_eig_ind = np.argsort(-values)
    return vectors[:,big_eig_ind[:self.n_clusters]]
      
  def _apply_constraints(self):
    self.aff_mat[self.ML[:,0], self.ML[:,1]] = 1
    self.aff_mat[self.CL[:,0], self.CL[:,1]] = 0  
