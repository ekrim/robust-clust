import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, \
                                   squareform
import sklearn.datasets as ds
from sklearn import svm
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
from sklearn.cluster import KMeans, \
                            SpectralClustering, \
                            MeanShift, \
                            AffinityPropagation, \
                            AgglomerativeClustering, \
                            DBSCAN

from .utils import find_errors, \
                   affinity, \
                   determine_relevance


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
  
def remove_constraints(data, constraint_mat, threshold=0.8, n_clusters=None, n_it=11):  
  n_cons = constraint_mat.shape[0]
  
  vote_mat = np.zeros((n_cons, n_cons)) > 0
  satisfied_constraints = np.zeros(n_cons) > 0
    
  CL = constraint_mat[constraint_mat[:,2] == 0, :2]
  ML = constraint_mat[constraint_mat[:,2] == 1, :2]
  for i in range(4):
    representation = data
    if i==0: 
      clus = KMeans(n_clusters=n_clusters)
    elif i==1:
      representation = squareform(affinity(pdist(data)))
      clus = SpectralClustering(n_clusters=n_clusters, affinity='precomputed')  
    elif i==2:
      clus = MeanShift()
    elif i==3:
      clus = AgglomerativeClustering(n_clusters=n_clusters, linkage='average')

    labs = clus.fit_predict(representation)
    vote_eligible = determine_relevance(constraint_mat, labs)

    conn_mat = labs[:,None] == labs[None,:]
    temp_sat = conn_mat[constraint_mat[:,0], constraint_mat[:,1]] == constraint_mat[:,2]  
    satisfied_constraints = satisfied_constraints | temp_sat
    vote_mat = vote_mat | (temp_sat[:,None] == temp_sat[None,:])  

  # Now we have determined which constraints are relvant
  # and which ones vote for one another
  # Mask the unsatisfied constraints, and the ones of same
  # type (ML or CL) from voting for each other
  vote_eligible *= satisfied_constraints[None,:]
  neq_mask = np.logical_xor(constraint_mat[:,2][:,None], constraint_mat[:,2][None,:]).astype(int)
  
  vote_eligible *= neq_mask
  vote_mat = vote_mat * vote_eligible.astype(int)
  
  trustedness = 0.5*np.ones(n_cons)
  trustedness[np.logical_not(satisfied_constraints)] = 0
  trust_mat = np.zeros((n_cons, n_it))
  trust_mat[:,0] = trustedness
  for i in range(1, n_it): 
    possible_trust_votes = np.dot(vote_eligible, trustedness)
    received_trust_votes = np.dot(vote_mat, trustedness).astype(float)
    trustedness = received_trust_votes / possible_trust_votes
    
    trustedness[possible_trust_votes == 0] = 1
    trust_mat[:,i] = trustedness

  keep_ind = trustedness >= threshold
  return keep_ind


class ConstraintsToLabels:
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
  label_set : array containing the labels for all the samples          
       contained in the array 'constrainedSamples'
       from the parent class
  """
  def __init__(self, classifier='svm'): 
    self.classifier = classifier
    self.hierarchical = AgglomerativeClustering(linkage='average')

  def fit_constrained(self, data):
    """Transform a set of pairwise constraints into a set of
    labeled training data
    """
    # Unsupervised hierarchical clustering
    self.hierarchical.fit(self.data)
    self.N,_ = self.data.shape
    
    self.label_set = np.arange(0,self.constrained_samps.size)
    # Find groups of constrained samples such that no CL 
    # constraints are found within a single group
    self.agglomerate_constrained_samples()
    
    # Merge the nodes representing agglomerated clusters of 
    # constrained samples to best obey ML and CL constraints 
    new_labels = self.merge_nodes()
    self.translate_labels(new_labels)

    # Use a simple kNN classifier with the converted pairwise
    # constraints to return a clustering for the data
    
    if self.classifier == 'knn':
      clf = KNeighborsClassifier(n_neighbors=1)
    elif self.classifier == 'svm':
      clf = svm.SVC(decision_function_shape='ovr')
    elif self.classifier == 'forest':
      clf = RandomForestClassifier(n_estimators=25)
    if np.unique(self.label_set).size==1:
      print(self.CL.shape[0], 'CL constraints, not enough')
    clf.fit(self.data[self.constrained_samps,:], self.label_set)
    self.labels = clf.predict(self.data)
    
  def merge_nodes(self):
    """We have an oversegmentation of the constrained points we wish
    to assign labels to, represented by the attribute self.label_set.
    This segmentation obeys CL constraints. Call each group of samples
    with the same labels a node. We find the net constraint value 
    between nodes, and group nodes to produce a proper label set for 
    the constrained samples.
    """
    uniqueLabels = np.unique(self.label_set)
    NuniqueLabels = uniqueLabels.size
    
    groupCenters = np.zeros((NuniqueLabels,self.data.shape[1]))
    groupPop = np.zeros((NuniqueLabels,1))
    # simMat contains the Nml - Ncl net constraint value
    self.simMat = np.zeros((NuniqueLabels,NuniqueLabels))
    # Loop over the node labels
    storeGroups = []
    for i in range(NuniqueLabels):
      group1 = self.constrained_samps[self.label_set==i]
      storeGroups += [group1]

      groupCenters[i,:] = np.mean(self.data[group1,:], axis=0)
      groupPop[i] = group1.size
      if i < (NuniqueLabels-1):
        # Loop over all other nodes
        for ii in range(i+1,NuniqueLabels):
          group2 = self.constrained_samps[self.label_set==ii]
          Ncl = self.number_of_constraints(group1,group2,0)
          Nml = self.number_of_constraints(group1,group2,1)
          val = Nml - Ncl
          self.simMat[i,ii] = val
          self.simMat[ii,i] = val
    #Nneigh = 2
    #if groupCenters.shape[0] > (Nneigh*2):
    #  self.complete_matrix(groupCenters, 2)

    #self.plot_graph_cut_problem(groupCenters, groupPop)
    
    centerDists = cc.squared_distance_matrix(groupCenters)
    return self.graph_cut_approx(centerDists, storeGroups)

  def graph_cut_approx(self, centerDists, storeGroups):
    """Find which of the nodes will merge based on the +/- constraint
    values between nodes. This is a very simple implementation, and will
    not produce a proper graph cut solution. Basically, we just merge
    two nodes if there is a net positive ML between them.
    """
    Nnodes = self.simMat.shape[0]
    new_labels = np.arange(Nnodes)
    Ngroups = Nnodes
    
    self.simMat += np.diag(-np.inf*np.ones(Nnodes))    
    keepLooping = True
    while keepLooping:
      maxInd = np.argmax(self.simMat) 
      r, c = np.unravel_index(maxInd, self.simMat.shape)
      maxVal = self.simMat[r,c]
      correctNclusters = False
      if self.n_clusters is not None:
        correctNclusters = self.n_clusters>=Ngroups    
      if maxVal > 0 and not correctNclusters:
        new_labels[new_labels==new_labels[c]] = new_labels[r]
        self.merge_columns(r, c)
        storeGroups[c] = np.append(storeGroups[c], storeGroups[r])
        Ngroups -= 1
      elif self.n_clusters is not None and maxVal >= 0:
        if maxVal==-np.inf:
          keepLooping = False
        elif Ngroups > self.n_clusters:
          if maxVal==0:
            r,c = self.break_tie_zero(centerDists)
          new_labels[new_labels==new_labels[c]] = new_labels[r]
          self.merge_columns(r, c)  
          storeGroups[c] = np.append(storeGroups[c], storeGroups[r])
          Ngroups -= 1  
        else:
          keepLooping = False
      else:
        keepLooping = False
    return new_labels
  
  def quick_hist(self, a):
    binEdges = np.append(np.unique(a), [np.inf])
    return np.histogram(a, bins=binEdges)
  
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
    cc.plot_labels(self.data[self.constrained_samps,:],self.label_set)
    
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

  def translate_labels(self, new_labels):
    """We have clustered nodes, so we need to reflect these changes
    in the label_set, which contains our generated labels for the
    constrained samples.
    """ 
    uniqueLabels = np.unique(self.label_set)
    for lab in uniqueLabels:
      self.label_set[self.label_set==lab] = new_labels[lab]

  def agglomerate_constrained_samples(self):
    """Given the merges of the hierarchical clustering, iterate 
    through them from the lowest level to the highest level. Make
    the labels of constrained samples the same if they are present
    in the same merge group with no CL constraints violated.
    """
    big_label_set = -np.ones((self.N,1))
    big_label_set = np.arange(self.N)
    #big_label_set[self.constrained_samps] = self.label_set.reshape((-1,1))

    all_merges = self.linkage_to_merges()  
    for merge in all_merges:  
      group1 = merge[0]
      group2 = merge[1]
      
      allSamps = np.append(group1, group2)  
      if not self.is_CL_violated(allSamps):  
        big_label_set[group1] = big_label_set[group2[0]]
        
    new_labels = big_label_set[self.constrained_samps]
    new_labels = cc.translate_to_counting_numbers(new_labels)
    self.label_set = new_labels  

  def linkage_to_merges(self):
    """Hierarchical clustering returns a matrix of 
    merges, starting with samples 0 to N-1, then 
    adding new labels for groups of merged samples.
    This is a generator function that returns 
    a list of the two arrays containing the indices 
    of all the samples in the two merged groups
    """
    self.merge_ind = []
    clus_mem = [np.asarray([x]) for x in range(self.N)]
    for i in range(0,self.N-1):
      group1 = self.hierarchical.children_[i,0]
      group2 = self.hierarchical.children_[i,1]
      clus_mem += [np.append( clus_mem[group1], clus_mem[group2])]
      yield [clus_mem[group1], clus_mem[group2]]    


if __name__ == '__main__':
  # Parameters---------------------------------------
  Nclusters = 4
  N = 500
  n_constraints = 500
  #---------------------------------------------------
  # Make some synthetic data
  data, labels = ds.make_blobs(n_samples=N, 
             n_features=2, 
                                     centers=Nclusters,
             center_box=(0,8))
  
  # Make some constaints  
  constraint_mat = cc.ConstrainedClustering.make_constraints(labels,
            n_constraints=n_constraints,
            errRate=0.0)
  ML = constraint_mat[constraint_mat[:,2]==1,:]
  CL = constraint_mat[constraint_mat[:,2]==0,:]
  constraint_mat = np.concatenate((ML[0:30,:], CL[:50,:]), axis=0)

  # Plot the data along with the constraints
  plt.figure()
  cc.ConstrainedClustering.plot_constraints(data, constraint_mat)
  plt.show()
  
  # Turn the pairwise constraints into labeled samples
  ctlObj = ConstraintsToLabels(data=data, 
             constraint_mat=constraint_mat, 
                                     n_clusters=Nclusters)
  ctlObj.fit_constrained()

  # Now these labels and their associated index can be used
  # in a classifier instead of clustering
  trainLabels = ctlObj.label_set
  trainInd = ctlObj.constrained_samps

  # Plot the resulting data, along with the training samples
  # that were produced from the pairwise constraints
  plt.figure()
  cc.plot_labels(data)
  cc.plot_labels(data[trainInd,:],trainLabels)  
  plt.show()

  #-------------------------
  N, Nclusters, n_constraints, errRate = (200,3,50,0.1)
  data, labels = ds.make_blobs(
    n_samples=N, 
    n_features=2, 
    centers=Nclusters
  )

  constraint_mat = cc.ConstrainedClustering.make_constraints(labels,
            n_constraints=n_constraints,
            errRate=errRate)  
  a = ImperfectOracles(data=data,
           constraint_mat=constraint_mat,
           n_clusters=Nclusters)
  keep_ind = a.remove_constraints()

  plt.figure()
  a.plot_removal(labels, keep_ind)
  plt.tight_layout()
  plt.show()
