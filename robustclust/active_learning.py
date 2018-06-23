import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as ds
from sklearn.cluster import AgglomerativeClustering

from . import constrained as cc


def make_constraints(labels, data=None, method='rand', num_constraints=None, errRate=0):
  N = labels.size

  # Half the number of samples is a good baseline
  num_constraints = N/2 if num_constraints is None else num_constraints

  if method == 'mmffqs':
    distMat = np.sqrt(squared_distance_matrix(data))
    queryMat, clusLabel = MMFFQS(labels, distMat, num_constraints)
    bigConstraintMat = all_pairwise(clusLabel)

  else:
    queryMat = np.random.randint(0,N,(num_constraints,2))

  queryMat = queryMat.astype(int)
  link = (labels[queryMat[:,0]] == labels[queryMat[:,1]])+0  
  # The samples whose link values we will invert
  errorInd = np.random.choice(2,num_constraints,p=[1-errRate,errRate]).astype('bool')  
  link = link.reshape((-1,1))
  link[errorInd,:] = 2 - np.power(2,link[errorInd,:])

  constraintMat = np.append(queryMat,link,axis=1)
  return constraintMat.astype(int), bigConstraintMat.astype(int)


def FFQS(labels, distMat, Nclass, Nconstraints):
  N = distMat.shape[0]
  nbrLabel = np.zeros(N)
  ind = np.arange(N)
  nbrLabel[np.random.random_integers(0,N-1,1)] = 1
  querCnt = 0
  constraintMat = np.zeros((Nconstraints,3))
  foundAll = False
  while querCnt < Nconstraints and (not foundAll):
    nbrInd = ind[nbrLabel > 0]
    candInd = ind[nbrLabel==0]
    block = distMat[nbrInd,:][:,candInd]
    minDist = np.min(block, axis=0)
    farInd = np.argmax(minDist)
    newPt = candInd[farInd]

    constraint = False
    nbrCnt = 1
    while (not constraint) and (nbrCnt <= np.max(nbrLabel)):
      thisHood = ind[nbrLabel==nbrCnt]
      constraint = labels[newPt]==labels[thisHood[0]]
      if querCnt < Nconstraints:
        constraintMat[querCnt,:] = [newPt, thisHood[0], constraint]
      querCnt += 1
      nbrCnt += 1
    if constraint:
      nbrLabel[newPt] = nbrCnt - 1
    else:  
      nbrLabel[newPt] = np.max(nbrLabel)+1
    uniqueNbr = np.setdiff1d(np.unique(nbrLabel),[0])
    if uniqueNbr.size==Nclass:
      foundAll = True
  return constraintMat, nbrLabel


def MMFFQS(labels, distMat, Nconstraints):
  Nclass = np.unique(labels).size
  N = distMat.shape[0]
  
  sortDist = np.sort(distMat, axis=1)
  kernel = 2*np.median(sortDist[:,2])
  simMat = np.exp(-distMat**2/(2*(kernel**2)))
  constraintMat, clusLabel = FFQS(labels, distMat, Nclass, Nconstraints)
  constraintMat.astype('int')  
  allInd = np.arange(N)
  exploreConstraints = constraintMat[constraintMat[:,0]!=0, 0:2].astype('int')
  skeletonInd = np.unique(exploreConstraints.reshape(-1))
  queryCnt = exploreConstraints.shape[0]

  clus = np.unique(np.setdiff1d(clusLabel,[0]))
  while queryCnt < Nconstraints:
    candidateInd = np.setdiff1d(allInd, skeletonInd)
    if candidateInd.size > 0:
      candSimToSkele = np.max(simMat[skeletonInd,:][:,candidateInd], axis=0)  
      qInd = np.argmin(candSimToSkele)
      q = candidateInd[qInd]
    else:
      q = np.random.random_integers(0,N-1,1)
    Nclus = clus.size
    simVec = np.zeros(Nclus)
    indVec = np.zeros(Nclus)
    for k in range(Nclus):
      ind_k = allInd[clusLabel==clus[k]]
      simInd = np.argmax(simMat[q, ind_k])
      simVec[k] = simMat[q, ind_k][simInd]
      indVec[k] = ind_k[simInd]
    sortInd = np.argsort(-simVec)
    indVec = indVec[sortInd]
    for k in range(Nclus):
      link = labels[q]==labels[indVec[k]]
      constraintMat[queryCnt,:] = [q, indVec[k], link]
      queryCnt += 1
      if link:
        clusLabel[q] = clusLabel[indVec[k]]
        break
      if k==Nclus:
        clusLabel[q] = np.max(clus) + 1
      if queryCnt==Nconstraints:
        break
    skeletonInd = np.append(skeletonInd, q)
  
  return constraintMat[:,0:2], clusLabel


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
    

if __name__=='__main__':
  N, Nclass, Nquery = (300, 6, 20)
  data, labels = ds.make_blobs(n_samples=N, n_features=2, centers=Nclass)
  a = ActiveClassDiscovery(data)
  trainInd = np.zeros(Nquery).astype(int)
  for i in range(Nquery):
    trainInd[i] = a.get_query()
    plt.figure()
    cc.plot_labels(data)
    cc.plot_labels(data[trainInd[:i+1]], labels[trainInd[:i+1]])
    plt.show()
