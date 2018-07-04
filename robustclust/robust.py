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
                   determine_relevance, \
                   plot_constraints, \
                   translate_to_counting_numbers


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

  clusterers_in_ensemble = ['kmeans', 'spectral', 'meanshift', 'agglom']

  n_cons = constraint_mat.shape[0]
  vote_mat = np.zeros((n_cons, n_cons)) > 0
  satisfied_constraints = np.zeros(n_cons) > 0
    
  CL = constraint_mat[constraint_mat[:,2] == 0, :2]
  ML = constraint_mat[constraint_mat[:,2] == 1, :2]

  for method in clusterers_in_ensemble:
    clus = get_clusterer(n_clusters, method)
     
    if method == 'spectral':
      representation = squareform(affinity(pdist(data)))
    else:
      representation = data

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
    """
    Args:
      classifier: one o

    """
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
    clf = get_classifier(self.classifier)

    if np.unique(self.label_set).size == 1:
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
    unique_labels = np.unique(self.label_set)
    n_unique_labels = unique_labels.size
    
    group_centers = np.zeros((n_unique_labels, self.data.shape[1]))
    group_pop = np.zeros((n_unique_labels, 1))
    # sim_mat contains the n_ml - n_cl net constraint value
    self.sim_mat = np.zeros((n_unique_labels, n_unique_labels))
    # Loop over the node labels
    store_groups = []
    for i in range(n_unique_labels):
      group1 = self.constrained_samps[self.label_set==i]
      store_groups += [group1]

      group_centers[i,:] = np.mean(self.data[group1,:], axis=0)
      group_pop[i] = group1.size
      if i < (n_unique_labels-1):
        # Loop over all other nodes
        for ii in range(i+1, n_unique_labels):
          group2 = self.constrained_samps[self.label_set==ii]
          n_cl = self.number_of_constraints(group1, group2, 0)
          n_ml = self.number_of_constraints(group1, group2, 1)
          val = n_ml - n_cl
          self.sim_mat[i,ii] = val
          self.sim_mat[ii,i] = val
    #n_neigh = 2
    #if group_centers.shape[0] > (n_neigh*2):
    #  self.complete_matrix(group_centers, 2)

    #self.plot_graph_cut_problem(group_centers, group_pop)
    
    center_dists = squareform(pdist(group_centers))
    return self.graph_cut_approx(center_dists, store_groups)

  def graph_cut_approx(self, center_dists, store_groups):
    """Find which of the nodes will merge based on the +/- constraint
    values between nodes. This is a very simple implementation, and will
    not produce a proper graph cut solution. Basically, we just merge
    two nodes if there is a net positive ML between them.
    """
    n_nodes = self.sim_mat.shape[0]
    new_labels = np.arange(n_nodes)
    n_groups = n_nodes
    
    self.sim_mat += np.diag(-np.inf*np.ones(n_nodes))    
    keep_looping = True
    while keep_looping:
      max_ind = np.argmax(self.sim_mat) 
      r, c = np.unravel_index(max_ind, self.sim_mat.shape)
      max_val = self.sim_mat[r,c]

      correct_n_clusters = False
      if self.n_clusters is not None:
        correct_n_clusters = self.n_clusters >= n_groups    

      if max_val > 0 and not correct_n_clusters:
        new_labels[new_labels==new_labels[c]] = new_labels[r]
        self.merge_columns(r, c)
        store_groups[c] = np.append(store_groups[c], store_groups[r])
        n_groups -= 1

      elif self.n_clusters is not None and max_val >= 0:
        if max_val == -np.inf:
          keep_looping = False

        elif n_groups > self.n_clusters:
          if max_val == 0:
            r,c = self.break_tie_zero(center_dists)

          new_labels[new_labels == new_labels[c]] = new_labels[r]
          self.merge_columns(r, c)  
          store_groups[c] = np.append(store_groups[c], store_groups[r])
          n_groups -= 1  

        else:
          keep_looping = False

      else:
        keep_looping = False

    return new_labels
  
  def quick_hist(self, a):
    bin_edges = np.append(np.unique(a), [np.inf])
    return np.histogram(a, bins=bin_edges)
  
  def break_tie_zero(self, center_dists):
    """This is just an approx because when two centers merge
    I am not adjusting the center_dists. If it doesn't work we
    can fix it
    """
    r,c = np.triu_indices(self.sim_mat.shape[0], 1)
    zero_ind = self.sim_mat[r,c]==0  
    row_zero = r[zero_ind]
    col_zero = c[zero_ind]
    zero_dists = center_dists[row_zero, col_zero]  
    best_pair = np.argmin(zero_dists)  
    return row_zero[best_pair], col_zero[best_pair]  
      
  def merge_columns(self, r, c):
    self.sim_mat[c,:] += self.sim_mat[r,:]
    self.sim_mat[:,c] = self.sim_mat[c,:]
    self.sim_mat[r,:] = -np.inf
    self.sim_mat[:,r] = -np.inf
    self.sim_mat[c,c] = -np.inf
    
  def plot_graph_cut_problem(self, centers, node_name):
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
    node_name - list containing the value that will be placed
         in each node 
    """
    plt.figure()
    unique_pairs = np.triu(self.sim_mat, k=1)
    row,col = np.nonzero(unique_pairs)    
    
    # Plot the constrained samples
    plot_constraints(self.data[self.constrained_samps,:], labels=self.label_set)
    
    # Plot lines between nodes representing the number and type
    # of constraints between them
    max_sim = np.max(np.abs(self.sim_mat[row, col]))
    for r,c in zip(row,col):

      if self.sim_mat[r,c] > 0:
        line_type = '-'
        line_color='b'
      else:  
        line_type = '--'
        line_color='r'

      line_thick = np.abs(self.sim_mat[r,c])
      line_thick *= 10/max_sim
      
      plt.plot(centers[[r,c],0], centers[[r,c],1],
        line_type, color=line_color, linewidth=line_thick)

    # Plot the nodes themselves
    plt.plot(centers[:,0], centers[:,1], 'o', 
      markersize=20, markerfacecolor=[0.7,0.7,0.7])
    
    # Put the population of the node in the center
    for i, val in enumerate(node_name):
      plt.text(centers[i,0], centers[i,1], str(val),
        horizontalalignment='center', verticalalignment='center') 

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
    n_nodes = self.sim_mat.shape[0]
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(centers)
    distances, indices = nbrs.kneighbors(centers)
    old_sim_mat = self.sim_mat.copy()

    for i in range(self.sim_mat.shape[0]):
      neigh_ind = np.zeros((1, n_nodes)) > 0
      neigh_ind[:, indices[i, 1:(k+1)]] = True
      neigh_ind = (neigh_ind & (old_sim_mat[i,:]==0)).flatten()
      self.sim_mat[i, neigh_ind] = 1
      self.sim_mat[neigh_ind, i] = 1      

  def translate_labels(self, new_labels):
    """We have clustered nodes, so we need to reflect these changes
    in the label_set, which contains our generated labels for the
    constrained samples.
    """ 
    unique_labels = np.unique(self.label_set)
    for lab in unique_labels:
      self.label_set[self.label_set == lab] = new_labels[lab]

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
      
      all_samps = np.append(group1, group2)  
      if not self.is_CL_violated(all_samps):  
        big_label_set[group1] = big_label_set[group2[0]]
        
    new_labels = big_label_set[self.constrained_samps]
    new_labels = translate_to_counting_numbers(new_labels)
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


def get_clusterer(n_clusters, method='kmeans'):
  clus = { 
    'kmeans': KMeans(n_clusters=n_clusters),
    'spectral': SpectralClustering(n_clusters=n_clusters, affinity='precomputed'),
    'meanshift': MeanShift(),
    'agglom': AgglomerativeClustering(n_clusters=n_clusters, linkage='average')}

  return clus[method]


def get_classifier(method='knn'):
  clf = {
    'knn': KNeighborsClassifier(n_neighbors=1),
    'svm': svm.SVC(decision_function_shape='ovr'),
    'forest': RandomForestClassifier(n_estimators=25)}
  
  return clf[method]
