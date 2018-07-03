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
          markerfacecolor=col, 
          markersize=marker_size)

  if constraint_mat is not None:
    for cons in constraint_mat:
      sampPair = cons[:2]
      if cons[2] == 1:
        lineType = '-'
      else:
        lineType = '--'
      plt.plot(data[sampPair,0], data[sampPair,1], lineType,
         color='black',
         linewidth=3)

  plt.show()
