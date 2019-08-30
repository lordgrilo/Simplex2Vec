#! usr/bin/python3

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import warnings
import json
import itertools
from tqdm import tqdm
import nibabel as nib

from scipy.spatial.distance import pdist, squareform
import itertools
from ast import literal_eval
import matplotlib.pyplot as plt

def read_simplex_json(path):

    with open(path, 'r') as fh:
        list_simplices = json.load(fh)


    list_simplices=[frozenset([str(i) for i in li]) for li in list_simplices]

    return list_simplices

def read_data_HCP(filename):

    img = nib.load(filename)              
    data = img.get_fdata()
    
    return data

def check_prediction(s2v,testSet):

    X = []
    X_labels = []
    X_baseNodes = []
    X_sizes = []
    for u in s2v.model.wv.vocab.keys():
        X.append(s2v.model[u])
        X_labels.append(u)
        X_baseNodes.append(literal_eval(X_labels[-1]))
        X_sizes.append(np.size(X_baseNodes[-1]))            

    sza = np.array(X_sizes)
    ns = len(sza)
    rns = np.arange(ns).astype('int')

    i3 = sza==3

    ik = sza==1
    nik = sum(ik)
    rns_ik = rns[ik]
    uk = np.array([X_baseNodes[r] for r in rns_ik])
    rnk = np.arange(nik).astype('int')

    dists = pdist([X[r] for r in rns_ik])
    dists = squareform(dists)

    # Measure perimeter of closed triplets
    dist_targ = []
    for ind in rns[i3]:
        u = X_baseNodes[ind]
        if len(u) !=3:
            raise Exception('Crazy 8s!')        
        for aa,bb in itertools.combinations(u, 2):                    
            ia = rnk[uk==aa]
            ib = rnk[uk==bb]
            dist_targ.append(dists[ia,ib])

    print('Histogram of perimeters from already closed triangle')
    print(np.histogram(np.concatenate(dist_targ)))
    
    # Measure perimeter of closed triplets
    dist_test = []
    for ts in testSet:
        u = np.array([int(t) for t in ts])
        if len(u) !=3:
            raise Exception('Crazy 8s!')        
        for aa,bb in itertools.combinations(u, 2):                    
            ia = rnk[uk==aa]
            ib = rnk[uk==bb]
            dist_test.append(dists[ia,ib])
            
    print('Histogram of perimeters from triangles yet to close')
    print(np.histogram(np.concatenate(dist_test)))

    # Measure perimeter of all triplets
    dist_control = []
    n_iters = 10000
    for ind in range(n_iters):
        u = np.random.choice(uk,size=3,replace=False)
        for aa,bb in itertools.combinations(u, 2):                    
            ia = rnk[uk==aa]
            ib = rnk[uk==bb]
            dist_control.append(dists[ia,ib])

    print('Histogram of preimeters of random node triangle')
    print(np.histogram(np.concatenate(dist_control)))

    ax = plt.subplot(111)
    plt.boxplot([dist_targ, dist_test, dist_control])
    ax.set_xticklabels(['Ideal','Test','Random'])
    ax.set_ylabel('Perimeter')
    plt.show()
    