import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as ds
from sklearn.cluster import AgglomerativeClustering

from .utils import pdist_idx, \
                   pdist_block, \
                   affinity, \
                   all_pairwise
from robustclust import plot_labels, \
                        plot_constraints


def get_constraints(labels, pdist_vec, method='rand', num_constraints=None, err_rate=0):
  N = labels.size

  # half the number of samples is a good baseline
  num_constraints = N/2 if num_constraints is None else num_constraints

  if method == 'mmffqs':
    query_mat, clus_label = MMFFQS(labels, pdist_vec, num_constraints)
    big_constraint_mat = all_pairwise(clus_label)

  if method == 'ffqs':
    query_mat, clus_label = FFQS(labels, pdist_vec, num_constraints)
    big_constraint_mat = all_pairwise(clus_label)

  elif method == 'rand':
    query_mat = np.random.randint(0, N, (num_constraints,2))
    big_constraint_mat = None
 
  else:
    assert False, 'no such method'

  query_mat = query_mat.astype(int)
  link = (labels[query_mat[:,0]] == labels[query_mat[:,1]])+0  

  # the samples whose link values we will invert
  error_ind = np.random.choice(2, num_constraints, p=[1-err_rate, err_rate]).astype('bool')  
  link = link.reshape((-1,1))
  link[error_ind,:] = 2 - np.power(2, link[error_ind,:])

  constraint_mat = np.append(query_mat,link,axis=1)
  return constraint_mat.astype(int), big_constraint_mat


def FFQS(labels, pdist_vec, num_constraints):
  """Furthest first query search
  """
  num_class = np.unique(labels).size
  N = labels.size

  nbr_label = np.zeros(N)
  ind = np.arange(N)
  nbr_label[np.random.random_integers(0, N-1, 1)] = 1
  quer_cnt = 0
  constraint_mat = np.zeros((num_constraints,3))
  found_all = False
  while quer_cnt < num_constraints and (not found_all):
    nbr_ind = ind[nbr_label > 0]
    cand_ind = ind[nbr_label == 0]
 
    block = pdist_block(pdist_vec, nbr_ind, cand_ind)

    min_dist = np.min(block, axis=0)
    far_ind = np.argmax(min_dist)
    new_pt = cand_ind[far_ind]

    constraint = False
    nbr_cnt = 1
    while (not constraint) and (nbr_cnt <= np.max(nbr_label)):
      this_hood = ind[nbr_label == nbr_cnt]
      constraint = labels[new_pt] == labels[this_hood[0]]
      if quer_cnt < num_constraints:
        constraint_mat[quer_cnt,:] = [new_pt, this_hood[0], constraint]
      quer_cnt += 1
      nbr_cnt += 1

    if constraint:
      nbr_label[new_pt] = nbr_cnt - 1
    else:  
      nbr_label[new_pt] = np.max(nbr_label) + 1

    unique_nbr = np.setdiff1d(np.unique(nbr_label),[0])
    if unique_nbr.size == num_class:
      found_all = True

  return constraint_mat, nbr_label


def MMFFQS(labels, pdist_vec, num_constraints):
  """minimax furthest first query search
  """
  num_class = np.unique(labels).size
  N = labels.size
  
  paff_vec = affinity(pdist_vec)

  constraint_mat, clus_label = FFQS(labels, pdist_vec, num_constraints)
  constraint_mat.astype('int')  
  all_ind = np.arange(N)
  explore_constraints = constraint_mat[constraint_mat[:,0]!=0, 0:2].astype('int')
  skeleton_ind = np.unique(explore_constraints.reshape(-1))
  query_cnt = explore_constraints.shape[0]

  clus = np.unique(np.setdiff1d(clus_label,[0]))
  while query_cnt < num_constraints:

    candidate_ind = np.setdiff1d(all_ind, skeleton_ind)
    if candidate_ind.size > 0:
      cand_sim_to_skele = np.max(pdist_block(paff_vec, skeleton_ind, candidate_ind), axis=0)  
      q_ind = np.argmin(cand_sim_to_skele)
      q = candidate_ind[q_ind]

    else:
      q = np.random.random_integers(0,N-1,1)

    num_clus = clus.size
    sim_vec = np.zeros(num_clus)
    ind_vec = np.zeros(num_clus).astype(int)
    for k in range(num_clus):
      ind_k = all_ind[clus_label == clus[k]]
      sim_ind = np.argmax(pdist_block(paff_vec, q, ind_k), axis=0)
      print('--------')
      print(q)
      print(ind_k)
      print(sim_ind)
      print('--------')
      sim_vec[k] = pdist_block(paff_vec, q, ind_k)[sim_ind]
      ind_vec[k] = ind_k[sim_ind]

    sort_ind = np.argsort(-sim_vec)
    ind_vec = ind_vec[sort_ind]
    for k in range(num_clus):
      print(q)
      print(k)
      print(ind_vec)
      link = labels[q] == labels[ind_vec[k]]
      constraint_mat[query_cnt,:] = [q, ind_vec[k], link]
      query_cnt += 1
      if link:
        clus_label[q] = clus_label[ind_vec[k]]
        break
      if k == num_clus:
        clus_label[q] = np.max(clus) + 1
      if query_cnt == num_constraints:
        break
    skeleton_ind = np.append(skeleton_ind, q)
  
  return constraint_mat[:,:2], clus_label


class ActiveClassDiscovery:
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
    

if __name__ == '__main__':
  N, num_class, Nquery = (300, 6, 20)
  data, labels = ds.make_blobs(n_samples=N, n_features=2, centers=num_class)
  
  constraint_mat, _ = get_constraints(labels, pdist(data), method='rand', num_constraints=20, err_rate=0)
  plot_constraints(data, constraint_mat)
