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
from scipy.stats import ks_2samp

def read_simplex_json(path):

    with open(path, 'r') as fh:
        list_simplices = json.load(fh)


    list_simplices=[frozenset([str(i) for i in li]) for li in list_simplices]

    return list_simplices

def read_data_HCP(filename):

    img = nib.load(filename)              
    data = img.get_fdata()
    
    return data

def get_open_closed_triangles(g, info = True):
    # Apparently a faster implementation
    closed_triangles = set([])
    open_triangles = set([])
    for u in tqdm_notebook(g.nodes()):
        neighbors = list(g.neighbors(u))
        for v, w in itertools.combinations(neighbors, 2):
            if g.has_edge(v,w):
                intersection_simplices = g[u][v]["simplices"].intersection(g[u][w]["simplices"])
                intersection_simplices = intersection_simplices.intersection(g[v][w]["simplices"])
                if len(intersection_simplices):
                    closed_triangles.add(frozenset((u,v,w)))
                else:
                    open_triangles.add(frozenset((u,v,w)))
    if info:
        print("Open: {}, Closed: {}".format(len(open_triangles), len(closed_triangles)))
    return open_triangles, closed_triangles

def check_prediction(s2v,testSet,metric='euclidean'):

    X = []
    X_labels = []
    X_baseNodes = []
    X_sizes = []
    for u in s2v.model.wv.vocab.keys():
        X.append(s2v.model[u])
        X_labels.append(u)
        X_baseNodes.append(literal_eval(X_labels[-1]))
        X_sizes.append(np.size(X_baseNodes[-1]))            

    # indexes all nodes
    sza = np.array(X_sizes)
    ns = len(sza)
    rns = np.arange(ns).astype('int')

    # indexes triplets
    i3 = sza==3
    rns_i3 = rns[i3]

    # indexes singletons
    ik = sza==1
    nik = sum(ik)
    rns_ik = rns[ik]
    rnk = np.arange(nik).astype('int')
    uk = np.array([X_baseNodes[r] for r in rns_ik])

    print('len of X is ' + str(len(X)))    
    print('len of X_ik is ' + str(len([X[r] for r in rns_ik])))    
    dists_ik = pdist([X[r] for r in rns_ik],metric=metric)
    dists_ik = squareform(dists_ik)
    print('shape of X dist is ' + str(dists_ik.shape))

    # Measure perimeter of closed triplets
    dist_targ = []
    for ind in rns_i3:
        u = X_baseNodes[ind]
        if len(u) !=3:
            raise Exception('Crazy 8s!')        
        temp = []
        for aa,bb in itertools.combinations(u, 2):                    
            ia = rnk[uk==aa]
            ib = rnk[uk==bb]
            temp.append(dists_ik[ia,ib])
        dist_targ.append(sum(temp))
    print('Histogram of perimeters from already closed triangle')
    print(np.histogram(np.concatenate(dist_targ)))
    
    # Measure perimeter of closed triplets
    dist_test = []
    for ts in testSet:
        u = np.array([int(t) for t in ts])
        if len(u) !=3:
            raise Exception('Crazy 8s!')        
        temp = []
        for aa,bb in itertools.combinations(u, 2):                    
            ia = rnk[uk==aa]
            ib = rnk[uk==bb]
            temp.append(dists_ik[ia,ib])
        dist_test.append(sum(temp))            
    print('Histogram of perimeters from triangles yet to close')
    print(np.histogram(np.concatenate(dist_test)))

    # Measure perimeter of all triplets
    dist_control = []
    n_iters = 10000
    for ind in range(n_iters):
        u = np.random.choice(uk,size=3,replace=False)
        temp = []
        for aa,bb in itertools.combinations(u, 2):                    
            ia = rnk[uk==aa]
            ib = rnk[uk==bb]
            temp.append(dists_ik[ia,ib])
        dist_control.append(sum(temp))
    print('Histogram of preimeters of random node triangle')
    print(np.histogram(np.concatenate(dist_control)))

    ax = plt.subplot(111)
    plt.boxplot([dist_targ, dist_test, dist_control])
    ax.set_xticklabels(['Ideal','Test','Random'])
    ax.set_ylabel('Perimeter')
    plt.show()
    
    stat, p = ks_2samp(np.concatenate(dist_test), np.concatenate(dist_control), alternative='less')    
    print('One-sided Kolmogorov-Smirnov statistic that test distribution is less than control, p value: ' + str(p))

    stat, p = ks_2samp(np.concatenate(dist_targ), np.concatenate(dist_control), alternative='less')    
    print('One-sided Kolmogorov-Smirnov statistic that ideal distribution is less than control, p value: ' + str(p))

def get_open_triangles(g, nodes, edges, triangles):
    open_triangles = set([])
    edges = set(edges)
    triangles = set(triangles)
    for u in tqdm(nodes):
        nei = list(g.neighbors(u))
        nei_nodes = list(nn.difference(u) for nn in nei)
        for n1, n2 in itertools.combinations(nei_nodes,2):
            if  set(n1 | n2) in edges:
                if not set(u | n1 | n2) in triangles:
                    open_triangles.add(u | n1 | n2)
            
    return list(open_triangles)

def check_trianglePromotion(s2v,testSet,controlSet,metric='cosine'):

    X = []
    X_labels = []
    X_baseNodes = []
    X_sizes = []
    for u in s2v.model.wv.vocab.keys():
        X.append(s2v.model[u])
        X_labels.append(u)
        X_baseNodes.append(literal_eval(X_labels[-1]))
        X_sizes.append(np.size(X_baseNodes[-1]))            

    # indexes all nodes
    sza = np.array(X_sizes)
    ns = len(sza)
    rns = np.arange(ns).astype('int')

    # indexes triplets
    i3 = sza==3
    rns_i3 = rns[i3]

    # indexes singletons
    ik = sza==1
    nik = sum(ik)
    rns_ik = rns[ik]
    rnk = np.arange(nik).astype('int')
    uk = np.array([X_baseNodes[r] for r in rns_ik])

    print('len of X is ' + str(len(X)))    
    print('len of X_ik is ' + str(len([X[r] for r in rns_ik])))    
    dists_ik = pdist([X[r] for r in rns_ik],metric=metric)
    dists_ik = squareform(dists_ik)
    print('shape of X dist is ' + str(dists_ik.shape))
    print('\n'+'Results trained on training set only.')
    print('Perimeters are taken wrt embedded nodes')

    # Measure perimeter of newly promoted triplets
    dist_targ = []
    for ind in rns_i3:
        u = X_baseNodes[ind]
        if len(u) !=3:
            raise Exception('Crazy 8s!')        
        temp = []
        for aa,bb in itertools.combinations(u, 2):                    
            ia = rnk[uk==aa]
            ib = rnk[uk==bb]
            temp.append(dists_ik[ia,ib])
        dist_targ.append(sum(temp))
    print('\n'+'Histogram of perimeters from already closed triangle')
    print(np.histogram(np.concatenate(dist_targ)))
    
    # Measure perimeter of newly closed triplets
    dist_test = []
    for ts in testSet:
        u = np.array([int(t) for t in ts])
        if len(u) !=3:
            raise Exception('Crazy 8s!')        
        temp = []
        for aa,bb in itertools.combinations(u, 2):                    
            ia = rnk[uk==aa]
            ib = rnk[uk==bb]
            temp.append(dists_ik[ia,ib])
        dist_test.append(sum(temp))            
    print('\n'+'Histogram of perimeters from triangles yet to be promoted to a 2-simplex')
    print(np.histogram(np.concatenate(dist_test)))

    # Measure perimeter of open triangles in training set
    dist_open = []
    for ts in controlSet:
        u = np.array([int(t) for t in ts])
        if len(u) !=3:
            raise Exception('Crazy 8s!')        
        temp = []
        for aa,bb in itertools.combinations(u, 2):                    
            ia = rnk[uk==aa]
            ib = rnk[uk==bb]
            temp.append(dists_ik[ia,ib])
        dist_open.append(sum(temp))            
    print('\n'+'Histogram of perimeters from all open triangles in training set')
    print(np.histogram(np.concatenate(dist_open)))
    
    # Measure perimeter of all triplets
    
    dist_random = []
    n_iters = 10000
    for ind in range(n_iters):
        u = np.random.choice(uk,size=3,replace=False)
        temp = []
        for aa,bb in itertools.combinations(u, 2):                    
            ia = rnk[uk==aa]
            ib = rnk[uk==bb]
            temp.append(dists_ik[ia,ib])
        dist_random.append(sum(temp))
    print('\n'+'Histogram of preimeters of random node triangles')
    print(np.histogram(np.concatenate(dist_random)))

    ax = plt.subplot(111)
    plt.boxplot([dist_targ, dist_test, dist_open, dist_random])
    ax.set_xticklabels(['Ideal','Test','Open','Random'])
    ax.set_ylabel('Perimeter')
    plt.show()
    
    stat, p = ks_2samp(np.concatenate(dist_targ), np.concatenate(dist_open), alternative='less')    
    print('\n'+'One-sided Kolmogorov-Smirnov statistic that ideal distribution is less than distribution from all open triangles, p value: ' + str(p))
    
    stat, p = ks_2samp(np.concatenate(dist_targ), np.concatenate(dist_random), alternative='less')    
    print('One-sided Kolmogorov-Smirnov statistic that ideal distribution is less than random node triangles, p value: ' + str(p))
    
    stat, p = ks_2samp(np.concatenate(dist_test), np.concatenate(dist_open), alternative='less')    
    print('One-sided Kolmogorov-Smirnov statistic that test distribution is less than distribution from all open triangles, p value: ' + str(p))

    stat, p = ks_2samp(np.concatenate(dist_test), np.concatenate(dist_random), alternative='less')    
    print('One-sided Kolmogorov-Smirnov statistic that test distribution is less than random node triangles, p value: ' + str(p))

    
def check_trianglePromotion_viaEdges(s2v,testSet,controlSet,metric='cosine'):

    X = []
    X_labels = []
    X_baseNodes = []
    X_sizes = []
    for u in s2v.model.wv.vocab.keys():
        X.append(s2v.model[u])
        X_labels.append(u)
        X_baseNodes.append(literal_eval(X_labels[-1]))
        X_sizes.append(np.size(X_baseNodes[-1]))            

    # indexes all nodes
    sza = np.array(X_sizes)
    ns = len(sza)
    rns = np.arange(ns).astype('int')

    # indexes triplets
    i3 = sza==3
    rns_i3 = rns[i3]
    
    # indexes doublets
    i2 = sza==2
    ni2 = sum(i2)
    rns_i2 = rns[i2]
    rn2 = np.arange(ni2).astype('int')
    u2 = np.array([X_baseNodes[r] for r in rns_i2])
    
    # indexes singletons
    ik = sza==1
    nik = sum(ik)
    rns_ik = rns[ik]
    rnk = np.arange(nik).astype('int')
    uk = np.array([X_baseNodes[r] for r in rns_ik])

    print('len of X is ' + str(len(X)))    
    print('len of X_ik is ' + str(len([X[r] for r in rns_ik])))    
    dists_ik = pdist([X[r] for r in rns_ik],metric=metric)
    dists_ik = squareform(dists_ik)
    print('shape of Xk dist is ' + str(dists_ik.shape))    
    print('len of X_i2 is ' + str(len([X[r] for r in rns_i2])))    
    dists_i2 = pdist([X[r] for r in rns_i2],metric=metric)
    dists_i2 = squareform(dists_i2)
    print('shape of X2 dist is ' + str(dists_i2.shape))
    print('\n'+'Results trained on training set only.')
    print('Perimeters are taken wrt embedded edges')

    # Measure perimeter of newly promoted triplets
    dist_targ = []
    for ind in rns_i3:
        u = X_baseNodes[ind]
        if len(u) !=3:
            raise Exception('Crazy 8s!')        
        temp = []
        twoList = list(itertools.combinations(u, 2))
        for aa,bb in itertools.combinations(twoList, 2):                    
            ia = rn2[np.all(u2==sorted(aa),axis=1)]
            ib = rn2[np.all(u2==sorted(bb),axis=1)]
            temp.append(dists_i2[ia,ib])
        dist_targ.append(sum(temp))
    print('\n'+'Histogram of perimeters from already closed triangle')
    print(np.histogram(np.concatenate(dist_targ)))
    
    # Measure perimeter of newly closed triplets
    dist_test = []
    for ts in testSet:
        u = np.array([int(t) for t in ts])
        if len(u) !=3:
            raise Exception('Crazy 8s!')        
        temp = []
        twoList = list(itertools.combinations(u, 2))        
        for aa,bb in itertools.combinations(twoList, 2):                    
            ia = rn2[np.all(u2==sorted(aa),axis=1)]
            ib = rn2[np.all(u2==sorted(bb),axis=1)]            
            temp.append(dists_i2[ia,ib])
            if temp[-1].size==0:
                print([ia,ib])
                raise Exception('struck')
        dist_test.append(sum(temp))            
    print('\n'+'Histogram of perimeters from triangles yet to be promoted to a 2-simplex')
    print(np.histogram(np.concatenate(dist_test)))

    # Measure perimeter of open triangles in training set
    dist_open = []
    for ts in controlSet:
        u = np.array([int(t) for t in ts])
        if len(u) !=3:
            raise Exception('Crazy 8s!')        
        temp = []
        twoList = list(itertools.combinations(u, 2))        
        for aa,bb in itertools.combinations(twoList, 2):                    
            ia = rn2[np.all(u2==sorted(aa),axis=1)]
            ib = rn2[np.all(u2==sorted(bb),axis=1)]
            temp.append(dists_i2[ia,ib])
        dist_open.append(sum(temp))            
    print('\n'+'Histogram of perimeters from all open triangles in training set')
    print(np.histogram(np.concatenate(dist_open)))
    
    ax = plt.subplot(111)

    plt.boxplot([dist_targ, dist_test, dist_open])
    ax.set_xticklabels(['Ideal','Test','Open'])
    ax.set_ylabel('Perimeter')
    plt.show()
    
    stat, p = ks_2samp(np.concatenate(dist_targ), np.concatenate(dist_open), alternative='less')    
    print('\n'+'One-sided Kolmogorov-Smirnov statistic that ideal distribution is less than distribution from all open triangles, p value: ' + str(p))
    
    stat, p = ks_2samp(np.concatenate(dist_test), np.concatenate(dist_open), alternative='less')    
    print('One-sided Kolmogorov-Smirnov statistic that test distribution is less than distribution from all open triangles, p value: ' + str(p))
