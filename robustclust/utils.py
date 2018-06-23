import numpy as np
import matplotlib.pyplot as plt
from scipy.special import binom
from scipy.spatial.distance import pdist


def pdist_block(pdist_vec, i, j):
  col_ind, row_ind = np.meshgrid(j, i)
  row_ind, col_ind = row_ind.flatten(), col_ind.flatten()
  return pdist_vec[pdist_idx(pdist_vec.size, row_ind, col_ind)].reshape((i.size, j.size))


def pdist_idx(N, i, j):
  """Get a indices for an array returned by pdist, so we
  can reference it as if it were the full square matrix.

    Args:
      - N: length of the pdist array, e.g. pdist(x).size
      - i: row index of matrix
      - j: col index of matrix

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
  a = np.asarray(a)
  b = np.asarray(b)
  memberInd = np.zeros_like(a)
  for element in b:
    memberInd[a==element] = 1
  return memberInd>0


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


def all_pairwise(labelSet):
  N = labelSet.size
  clusList = np.setdiff1d(np.unique(labelSet), [0])
  allInd = np.arange(N)
  allConstrained = allInd[labelSet>0]
  bigConstraintMat = np.zeros((0,3))
  for i in clusList:
    thisClus = allInd[labelSet==i]
    otherClus = np.setdiff1d(allConstrained, thisClus)
    x, y = np.meshgrid(thisClus, thisClus)
    x = x.reshape((-1,1))
    y = y.reshape((-1,1))
    mlBlock = np.concatenate((x,y,np.ones(x.shape)), axis=1)
    x, y = np.meshgrid(thisClus, otherClus)
    x = x.reshape((-1,1))
    y = y.reshape((-1,1))
    clBlock = np.concatenate((x,y,np.zeros(x.shape)), axis=1)
    bigConstraintMat = np.concatenate((bigConstraintMat,mlBlock,clBlock), axis=0)
  return bigConstraintMat


def plot_labels(data, labels=None):
  """Plot the data colored according to the unique class 
  labels. If no labels are provided, the data is grey.
  """
  if labels is not None:
    classes = set(labels)
    colors = plt.cm.Spectral(np.linspace(0,1,len(classes)))
    for lab, col in zip(classes, colors):
      ind = labels == lab
      plt.plot(data[ind,0], data[ind,1], 'o', 
          markerfacecolor=col, 
          markersize=10)
  else:
    plt.plot(data[:,0], data[:,1], 'o',
       markerfacecolor=[0.7, 0.7, 0.7],
       markersize=10)


def plot_constraints(data, constraint_mat):
  """Plot the data (all grey) and the pairwise 
  constraints

  ML constraints will be solid lines, while CL 
  constraints will be dashed lines
  """    
  plt.plot(data[:,0],data[:,1],'o',
     markerfacecolor=[0.7,0.7,0.7],
     markersize=5)
  for cons in constraint_mat:
    sampPair = cons[:2]
    if cons[2] == 1:
      lineType = '-'
    else:
      lineType = '--'
    plt.plot(data[sampPair,0], data[sampPair,1], lineType,
       color='black',
       linewidth=3)
