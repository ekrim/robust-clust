import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import sklearn.datasets as ds
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans

import constrained_clustering as cc
import constraints_to_labels as ctl


Ntrial = 10
activeLearning = 'mmffqs'
saveFile = activeLearning+str(Ntrial)
consFracVec = np.linspace(0.1,1,9)
#consVec = [50,100,150,200,250]
errRate = 0.0

#balance, breast
datasetList = ['ecoli','glass','ionosphere',
         'iris','parkinsons','segmentation','wine']
# Leave out: yeast, sonar
plt.figure()

ariMat = np.zeros((len(consFracVec), 2, len(datasetList)))
for iData, fileName in enumerate(datasetList):
  print(fileName)
  df = pd.read_csv(fileName+'.csv', header=None)
  bigMat = df.as_matrix()
  data = bigMat[:,:-1]
  labels = bigMat[:,-1]

  N = data.shape[0]
  Nclusters = np.unique(labels).size
  
  kmeans = KMeans(n_clusters=Nclusters)
  kmeans.fit(data)
  baseline = nmi(labels, kmeans.labels_)
  for iCons, consFrac in enumerate(consFracVec):
    Nconstraints = np.round(N*consFrac)
 
    ariVec = np.zeros(2)
    for it in range(Ntrial):  
      constraintMat, bigConstraintMat = cc.ConstrainedClustering.make_constraints(
        labels, 
        data=data,
        method=activeLearning,
        Nconstraints=Nconstraints,
        errRate=errRate
      )
    
      # Constraints-to-labels
      ctlObj = ctl.ConstraintsToLabels(
        classifier='svm',
        data=data,
        constraintMat=bigConstraintMat,
        n_clusters=Nclusters
      )
      ctlObj.fit_constrained()
      ctlLabels = ctlObj.labels
      ariVec[0] += nmi(labels, ctlLabels)

      # E2CP
      e2cp = cc.E2CP(
        data=data, 
        constraintMat=bigConstraintMat, 
        n_clusters=Nclusters
      )
      e2cp.fit_constrained()
      e2cpLabels = e2cp.labels
      ariVec[1] += nmi(labels, e2cpLabels)
    
    ariMat[iCons,:,iData] = ariVec/Ntrial
    
  plt.subplot(3,3,iData+1)
  plt.plot(consFracVec, ariMat[:,0,iData], 'b-o', label='CtL')
  plt.plot(consFracVec, ariMat[:,1,iData], 'r-^', label='E2CP')
  plt.plot(consFracVec, baseline*np.ones(consFracVec.shape), 'k--', label='k-means', linewidth=1)
  plt.title(fileName)
  plt.ylabel('NMI', fontsize=12)
  if iData==7:
    plt.xlabel('Number of constraints as fraction of N', fontsize=14)
    plt.legend(loc=4)
matplotlib.rc('xtick', labelsize=10)
matplotlib.rc('ytick', labelsize=10)  
plt.tight_layout()
plt.show()
print(ariMat)
np.save(saveFile, ariMat)  
      
