import numpy as np
import matplotlib.pyplot as plt
from scipy.special import binom
from scipy.spatial.distance import pdist


def pdist_block(pdist_vec, i, j):
  """Slice the pdist ndarray as if it were a squareform
  matrix. 
  
  Args:
    pdist_vec: ndarray output from pdist()
    i: ndarray row index of matrix
    j: ndarray col index
    
  Returns:
    (i.size, j.size) ndarray from distance matrix
  """
  col_ind, row_ind = np.meshgrid(j, i)
  row_ind, col_ind = row_ind.flatten(), col_ind.flatten()
  return pdist_vec[pdist_idx(pdist_vec.size, row_ind, col_ind)].reshape((i.size, j.size))


def pdist_idx(N, i, j):
  """Get indices for an ndarray returned by pdist, so we
  can reference it as if it were the full square matrix.

  Args:
    N: length of the pdist array, e.g. pdist(x).size
    i: ndarray row index of matrix
    j: ndarray col index of matrix

  Returns:
    ndarray consisting of one pdist index per each i,j pair

  """
  D = int((1+np.sqrt(1+8*N))/2)
  if type(i) is int:
    i = np.array([i])
    j = np.array([j])

  idx_where_flip = i > j
  i[idx_where_flip], j[idx_where_flip] = j[idx_where_flip], i[idx_where_flip]
  
  unique_i = np.unique(i)
  fib_vals = binom(unique_i + 2, 2)
  fib_dict = dict(zip(unique_i, fib_vals))   
  
  return (D * i + j - np.array([fib_dict[val] for val in i])).astype(int)
  

def affinity(pdist_vec):
  med_dist = np.median(pdist_vec)
  kernel_size = 3 * med_dist
  return np.exp(-pdist_vec/(2*kernel_size**2))


def ismember(a, b):
  """Return an array with the same size of 'a' that 
  indicates if an element of 'a' is in 'b'. Can be 
  used as an index for slicing since it contains bool 
  elements
  """
  a = np.array(a)
  b = np.array(b)
  member_ind = np.zeros_like(a)
  for element in b:
    member_ind[a == element] = 1
  return member_ind>0


def translate_to_counting_numbers(lab_orig):
  """Perform a 1-to-1 mapping of the numbers from a numerical
  array to a new array with elements that are in the set
  {0,1,...,M-1} where M is the number of unique elements in 
  the original array 
  """
  lab_orig = np.array(lab_orig)
  lab_unique = np.unique(a)
  d = dict(zip(lab_unique, np.arange(lab_unique.size)))   
  return np.array([d[val] for val in lab_orig]).astype(int)


def all_pairwise(label_set):
  N = label_set.size
  clus_list = np.setdiff1d(np.unique(label_set), [0])
  all_ind = np.arange(N)
  all_constrained = all_ind[label_set>0]
  big_constraint_mat = np.zeros((0,3))
  for i in clus_list:
    this_clus = all_ind[label_set==i]
    other_clus = np.setdiff1d(all_constrained, this_clus)

    x, y = np.meshgrid(this_clus, this_clus)
    x, y = x.reshape((-1,1)), y.reshape((-1,1))
    ml_block = np.concatenate((x, y, np.ones(x.shape)), axis=1)

    x, y = np.meshgrid(this_clus, other_clus)
    x, y  = x.reshape((-1,1)), y.reshape((-1,1))
    cl_block = np.concatenate((x, y, np.zeros(x.shape)), axis=1)

    big_constraint_mat = np.concatenate((big_constraint_mat, ml_block, cl_block), axis=0)
  return big_constraint_mat


def determine_relevance(cons_mat, labs):
  vote_eligible = np.zeros((cons_mat.shape[0], cons_mat.shape[0])).astype(int)
  for i, cons in enumerate(cons_mat):
    clus1, clus2 = labs[cons[0]], labs[cons[1]]
    col1 = labs[cons_mat[:,0]][:,None]
    col2 = labs[cons_mat[:,1]][:,None]
    labs_of_cons = np.concatenate([col1, col2], axis=1)

    n_matches1 = np.sum(labs_of_cons == np.asarray([clus1, clus2])[None,:], axis=1)
    n_matches2 = np.sum(labs_of_cons == np.asarray([clus2, clus1])[None,:], axis=1)
    n_matches = np.maximum(n_matches1, n_matches2)

    if cons[2] == 1:
      valid_ind = n_matches == 2
    else:
      valid_ind = n_matches >= 1

    vote_eligible[i, valid_ind] = 1    

  return vote_eligible


def find_errors(labels, constraint_mat):
  true_conn_mat = (labels[:,None] == labels[None,:]).astype(int)
  true_cons = true_conn_mat[constraint_mat[:,0], constraint_mat[:,1]]
  err_ind = true_cons != constraint_mat[:,2]
  return err_ind


def plot_removal(data, labels, keep_ind, constraint_mat):
  err_ind = find_errors(labels, constraint_mat)

  # The ones we kept
  plt.subplot(2,2,1)
  plot_constraints(data, labels=labels)
  plt.title('True grouping')    

  # Errors
  plt.subplot(2,2,2)
  plot_constraints(data, constraint_mat=constraint_mat[err_ind,:])
  plt.title('Oracle errors')
  
  # Good ones we removed
  plt.subplot(2,2,3)
  toss_ind = np.logical_not(keep_ind)
  correct_ind = np.logical_not(err_ind)
  correct_tossed = np.logical_and(toss_ind, correct_ind)
  plot_constraints(data, constraint_mat=constraint_mat[correct_tossed,:])
  plt.title('Correct constraints removed')    

  # Errors we left
  plt.subplot(2,2,4)
  plot_constraints(data, constraint_mat=constraint_mat[np.logical_and(keep_ind, err_ind),:])  
  plt.title('Errors remaining')


def plot_constraints(data, labels=None, constraint_mat=None):
  """Plot the data and possibly labels and/or pairwise constraints. 
  ML constraints will be solid lines, while CL constraints will be 
  dashed lines.

    Args:
      data: (N,D) ndarray of the data
      labels: array of class labels
      constraint_mat: (num_constraints, 3) ndarray, where each row
                      is (index of sample 1, index of sample 2, 
                      constraint type), where constraint type is 1
                      for must-link, -1 for cannot-link

  """    
  marker_size = 7
  if labels is None:
    plt.plot(data[:,0], data[:,1], 'o',
       markerfacecolor=[0.7, 0.7, 0.7],
       markersize=marker_size)

  else: 
    classes = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(classes)))
    for lab, col in zip(classes, colors):
      ind = labels == lab
      plt.plot(data[ind, 0], data[ind, 1], 'o', 
        markerfacecolor=col, markersize=marker_size)

  if constraint_mat is not None:
    for cons in constraint_mat:
      sampPair = cons[:2]
      if cons[2] == 1:
        lineType = '-'
      else:
        lineType = '--'
      plt.plot(data[sampPair,0], data[sampPair,1], 
         lineType, color='black', linewidth=3)

  plt.show()
