#! /usr/bin/pyhton3

import networkx as nx
import numpy as np
import warnings
import itertools
from tqdm import tqdm
from scipy.special import perm, factorial
import os

def _stirling(n):
    return np.ceil(np.sqrt(2*np.pi*n)*(n/np.e)**n)

def simplex2hasse_uniform(data, max_order=None):
    '''Returns the Hasse diagram as a networkX graph (undirected, simple). 
    Input: list of frozensets
    Output: networkx Graph object
    '''
    
    def _build_simplices(simplex, l):
        #recursive function that calculates all possible simplices
        for s in itertools.combinations(simplex, len(simplex)-1):
            
            l.append((frozenset(simplex),frozenset(s)))
            
            if len(s)>1:
                _build_simplices(s,l)
        return 


    # execute cleaning of the dataset - remove duplicate simplices
    data = list(set(data))

    # initialize the Hasse graph (diagram)
    g = nx.Graph()

    # go through the simplices, create nodes and edges
    for u in tqdm(data, 'Creating Hasse diagram'):
        if None == max_order or (len(u)<max_order+1 and len(u) >= 1):
            buff = []
            _build_simplices(u, buff)
            g.add_edges_from(buff)
        else:
            for v in itertools.combinations(u,max_order+1):
                buff = []
                _build_simplices(v, buff)
                g.add_edges_from(buff)
    
    weight = 1
    nx.set_node_attributes(g,weight,'weight')
    
    return g


def simplex2hasse_counts(data, max_order=None):
    '''Returns the Hasse diagram as a networkX graph (undirected, weighted). Only simplices appearing in the dataset
    receive the non-trivial weight that equals to the number of their appearance. Other simplices receive small weight epsilon,
    thus eliminating zero probability nodes. 
    Input: list of frozensets
    Output: networkx Graph object
    '''

    epsilon = 1e-15
       
    def _build_simplices(simplex, l):
        #recursive function that calculates all possible son simplices
        for s in itertools.combinations(simplex, len(simplex)-1):
            fs = frozenset(s)
            if fs not in weights_dict:
                weights_dict[fs] = epsilon
            l.append((frozenset(simplex), fs))
            if len(s)>1:
                _build_simplices(s,l)
        return  
    
    
    # execute cleaning of the dataset - remove duplicate simplices
    data = list(set(data))

    # initialize the Hasse graph (diagram)
    g = nx.Graph()
    weights_dict = {}
    
    # go through the simplices, create nodes
    for u in tqdm(data, 'Creating Hasse diagram'):
        if None == max_order or (len(u) < max_order+1 and len(u) >= 1):
            if u not in weights_dict:
                weights_dict[u] = 1.
            else:
                weights_dict[u] += 1.
            buff = []
            _build_simplices(u, buff)
            g.add_edges_from(buff)
        else:
            for v in itertools.combinations(u, max_order+1):
                if frozenset(v) not in weights_dict:
                    weights_dict[frozenset(v)] = 1.
                else:
                    weights_dict[frozenset(v)] += 1.
                buff = []
                _build_simplices(v, buff)
                g.add_edges_from(buff)
    
    nx.set_node_attributes(g, weights_dict, 'weight')
    
    return g



def simplex2hasse_LO(data, max_order=None):
    '''Returns the Hasse diagram as a networkX graph (undirected, weighted) with cumulative appearance counts on nodes. 
    When a n-simplex appears in the data, n-simplex receives weight 1, (n-1)-simplex receives .....
    Input: list of frozensets
    Output: networkx Graph object
    '''
        
    def _build_simplices(simplex, l):
        #recursive function that calculates all possible son simplices
        for s in itertools.combinations(simplex, len(simplex)-1):
            fs = frozenset(s)
            if len(s) > 0:
                if fs in weights_dict:
                    weights_dict[fs] += perm(top_simplex_order, level)/(_stirling(level))
                else:
                    weights_dict[fs] = perm(top_simplex_order, level)/(_stirling(level))
                l.append((frozenset(simplex), fs))
            if len(s) > 1:
                _build_simplices(s,l)
        return  
    
    
    # execute cleaning of the dataset - remove duplicate simplices
    data = list(set(data))

    # initialize the Hasse graph (diagram)
    g = nx.Graph()
    weights_dict = {}
    
    # go through the simplices, create nodes
    for u in tqdm(data, 'Creating Hasse diagram'):

        if None == max_order or (len(u) < max_order+1 and len(u) >= 1):
            if u not in weights_dict:
                weights_dict[u] = 1.
            else:
                weights_dict[u] += 1.
            buff = []
            _build_simplices(u, buff)
            g.add_edges_from(buff)
        else:
            for v in itertools.combinations(u, max_order+1):
                if frozenset(v) not in weights_dict:
                    weights_dict[frozenset(v)] = 1.
                else:
                    weights_dict[frozenset(v)] += 1.
                buff = []
                _build_simplices(v, buff)
                g.add_edges_from(buff)
    
    nx.set_node_attributes(g, weights_dict, 'weight')
    
    return g

def simplex2hasse_LOadjusted(data, max_order=None):
    '''Returns the Hasse diagram as a networkX graph (undirected, weighted) with cumulative appearance counts on nodes 
    adjusted by the diagram level. Adjustment coefficient for n-simplex on level k (level of k-simplices) is 1/((n+1)*n*..*(k+1))
    
    Example: 3-simplex (tetrahedron) appearing in the data receives weight 1, adjacent 2-simplices (triangles) receive weigth 1/4, 
        1-simplices (edges) receive 1/(4*3), 0-simplices (nodes) receive 1/(4*3*2).
    Input: list of frozensets
    Output: networkx Graph object
    '''
        
    def _build_simplices(simplex, l):
        #recursive function that calculates all possible son simplices
        for s in itertools.combinations(simplex, len(simplex)-1):
            fs = frozenset(s)
            level = top_simplex_order-len(simplex) + 1
            if len(s) > 0:
                
                ###
                if fs in weights_dict:
                    weights_dict[fs] += 1/(_stirling(level)*_stirling(level+1))
                else:
                    weights_dict[fs] = 1/(_stirling(level)*_stirling(level+1))

                ## EXACT CALCULATIONS : Comment prev bloc and uncomment this one
#                 if fs in weights_dict:
#                     weights_dict[fs] += 1/(factorial(level)*factorial(level+1)) #perm(top_simplex_order, level))
#                 else:
#                     weights_dict[fs] = 1/(factorial(level)*factorial(level+1)) #perm(top_simplex_order, level))


            l.add((frozenset(simplex), fs))
            if len(s) > 1:
                _build_simplices(s,l)
        return  
    
    
    # execute cleaning of the dataset - remove duplicate simplices
    data = list(set(data))

    # initialize the Hasse graph (diagram)
    g = nx.Graph()
    weights_dict = {}
    
    # go through the simplices, create nodes
    for u in tqdm(data, 'Creating Hasse diagram'):

        if None == max_order or (len(u) < max_order+1 and len(u) >= 1):
            if u not in weights_dict:
                weights_dict[u] = 1.
            else:
                weights_dict[u] += 1.
            buff = set({})
            top_simplex_order = len(u)
            _build_simplices(u, buff)
            g.add_edges_from(buff)
        else:
            for v in itertools.combinations(u, max_order+1):
                if frozenset(v) not in weights_dict:
                    weights_dict[frozenset(v)] = 1.
                else:
                    weights_dict[frozenset(v)] += 1.
                buff = set({})
                top_simplex_order = len(v)
                _build_simplices(v, buff)
                g.add_edges_from(buff)

    
    nx.set_node_attributes(g, weights_dict, 'weight')
    return g

def simplex2hasse_LOlinear(data, max_order=None):
    '''Returns the Hasse diagram as a networkX graph (undirected, weighted) with cumulative appearance counts on nodes 
    adjusted by the diagram level. Adjustment coefficient for n-simplex on level k (level of k-simplices) is 1/(k+1))
    
    Example: 3-simplex (tetrahedron) appearing in the data receives weight 1, adjacent 2-simplices (triangles) receive weigth 1/2, 
        1-simplices (edges) receive 1/3, 0-simplices (nodes) receive 1/4.
    Input: list of frozensets
    Output: networkx Graph object
    '''
        
    def _build_simplices(simplex, l):
        #recursive function that calculates all possible son simplices
        for s in itertools.combinations(simplex, len(simplex)-1):
            fs = frozenset(s)
            level = top_simplex_order-len(simplex) + 1
            if len(s) > 0:
                
                if fs in weights_dict:
                    weights_dict[fs] += 1/(_stirling(level)*(level+1))
                else:
                    weights_dict[fs] = 1/(_stirling(level)*(level+1))

            l.add((frozenset(simplex), fs))
            if len(s) > 1:
                _build_simplices(s,l)
        return  
    
    
    # execute cleaning of the dataset - remove duplicate simplices
    data = list(set(data))

    # initialize the Hasse graph (diagram)
    g = nx.Graph()
    weights_dict = {}
    
    # go through the simplices, create nodes
    for u in tqdm(data, 'Creating Hasse diagram'):

        if None == max_order or (len(u) < max_order+1 and len(u) >= 1):
            if u not in weights_dict:
                weights_dict[u] = 1.
            else:
                weights_dict[u] += 1.
            buff = set({})
            top_simplex_order = len(u)
            _build_simplices(u, buff)
            g.add_edges_from(buff)
        else:
            for v in itertools.combinations(u, max_order+1):
                if frozenset(v) not in weights_dict:
                    weights_dict[frozenset(v)] = 1.
                else:
                    weights_dict[frozenset(v)] += 1.
                buff = set({})
                top_simplex_order = len(v)
                _build_simplices(v, buff)
                g.add_edges_from(buff)
    
    nx.set_node_attributes(g, weights_dict, 'weight')
    
    return g

def simplex2hasse_proportional(data, max_order=None):
    '''Returns the Hasse diagram as a networkX DiGraph (directed, weighted) with cumulative appearance counts on nodes 
    adjusted by the diagram level. Adjustment coefficient for n-simplex on level k (level of k-simplices) is 1/(k+1))
    
    Example: 3-simplex (tetrahedron) appearing in the data receives weight 1, adjacent 2-simplices (triangles) receive weigth 1/2, 
        1-simplices (edges) receive 1/3, 0-simplices (nodes) receive 1/4.
    Input: list of frozensets
    Output: networkx Graph object
    '''
        
    def _build_simplices(simplex, l):
        #recursive function that calculates all possible son simplices
        for s in itertools.combinations(simplex, len(simplex)-1):
            fs = frozenset(s)
            level = top_simplex_order-len(simplex) + 1
            if len(s) > 0:
                
                if fs in weights_dict:
                    weights_dict[fs] += 1/(_stirling(level)*(level+1))
                else:
                    weights_dict[fs] = 1/(_stirling(level)*(level+1))

            l.add((frozenset(simplex), fs))
            if len(s) > 1:
                _build_simplices(s,l)
        return  
        
    # execute cleaning of the dataset - remove duplicate simplices
    data = list(set(data))

    # initialize the Hasse graph (diagram)
    g = nx.Graph()
    weights_dict = {}
    
    # go through the simplices, create nodes
    for u in tqdm(data, 'Creating Hasse diagram'):

        if None == max_order or (len(u) < max_order+1 and len(u) >= 1):
            if u not in weights_dict:
                weights_dict[u] = 1.
            else:
                weights_dict[u] += 1.
            buff = set({})
            top_simplex_order = len(u)
            _build_simplices(u, buff)
            g.add_edges_from(buff)
        else:
            for v in itertools.combinations(u, max_order+1):
                if frozenset(v) not in weights_dict:
                    weights_dict[frozenset(v)] = 1.
                else:
                    weights_dict[frozenset(v)] += 1.
                buff = set({})
                top_simplex_order = len(v)
                _build_simplices(v, buff)
                g.add_edges_from(buff)
    
    nx.set_node_attributes(g, weights_dict, 'prob')
    all_cliques = g.nodes
    bg = nx.DiGraph()
    
    for u in g.nodes:
        sz = len(u)
        hi_sz = sz+1
        probs_hi = 0
        lo_sz = sz-1
        probs_lo = 0
        for nb in g.neighbors(u):
            if len(nb)==hi_sz:
                probs_hi += g.nodes[nb]['prob']
            elif len(nb)==lo_sz:
                probs_lo += g.nodes[nb]['prob']
        for nb in g.neighbors(u):
            if len(nb)==hi_sz:
                if probs_lo > 0:
                    bg.add_edge(u, nb, weight=g.nodes[nb]['prob']/probs_hi/2)                
                else:
                    bg.add_edge(u, nb, weight=g.nodes[nb]['prob']/probs_hi)                    
            elif len(nb)==lo_sz:
                if probs_hi > 0:
                    bg.add_edge(u, nb, weight=g.nodes[nb]['prob']/probs_lo/2)                    
                else:
                    bg.add_edge(u, nb, weight=g.nodes[nb]['prob']/probs_lo)                    
        
    return bg

def graph2hasse(M, threshold, maxOrder):
    #graph2hasse chooses a cutoff threshold to establish simplices    
    
    # M is an MxM undirected weighted graph
    # Values should be Fischer z-scored to achieve a Gaussian distribution
    
    # minZStat is a lower bound for choice of cutoff. Must be reasonable for computation and interpretation
    
    # Perform thresholding
    Mt = nx.convert_matrix.to_numpy_matrix(M)
    cutoff = threshold
    #print('cutoff = ' + str(cutoff))
    Mt[Mt<cutoff] = 0
    Mt[Mt>cutoff] = 1

    Mx=nx.from_numpy_matrix(Mt)
    
    all_cliques = [np.asarray(sorted(l)) for l in nx.enumerate_all_cliques(Mx)]
    all_cliques_str = [str(sorted(l)) for l in nx.enumerate_all_cliques(Mx)]
    sz_all_cliques_str = np.size(all_cliques_str)
    
    mx_cliques = [np.asarray(sorted(l)) for l in nx.find_cliques(Mx)]
    mx_cliques_str = [str(sorted(l)) for l in nx.find_cliques(Mx)]
    sz_mx_cliques_str = np.size(mx_cliques_str)
    
    # initialize the Hasse graph (diagram)
    hasse = nx.Graph()

    # go through the simplices, create nodes
    #sz_l = []
    for i in range(sz_all_cliques_str):
        u = all_cliques_str[i]
        v = all_cliques[i]
        sz = v.size
        #sz_l.append(sz)
        prb = 1
        hasse.add_node(u,size=sz,bNodes=v,prob=prb,name=u)          
        
    #plt.hist(sz_l)
    #plt.suptitle(['Cutoff = ' + str(cutoff)])
    #plt.show()
    
    def deeperEdge(ss, ll, hasse):
        v = hasse.nodes[ss]['bNodes']
        sz = hasse.nodes[ss]['size']
        for vv in itertools.combinations(v,sz-1):
            vvv = str(list(vv))
            ll.append((ss,vvv))
            
            if sz>2:
                ll = deeperEdge(vvv,ll,hasse)        
                
        return ll
        
    # create edges in the Hasse graph (diagram)
    # Initialize Hess    
    for i in np.arange(sz_mx_cliques_str):
        u = mx_cliques_str[i]
        sz = hasse.nodes[u]['size']
        if sz>1:
            buff = []
            buff = deeperEdge(u,buff,hasse)
            hasse.add_edges_from(buff)                    
            
    return hasse #, cutoff, sz_l

def graph2hasse_proportional(M, threshold, maxOrder):
    #graph2hasse chooses a cutoff threshold to establish simplices    
    
    # M is an MxM undirected weighted graph of gaussian distributed correlation z-statistics           
    # threshold is an lower bound for choice of cutoff. Must be reasonable for computation and interpretation
    # Output graph, hasse, is directed with weights proportional to the inverse of the sum of input edge weights
    
    # Perform thresholding
    n, n = M.shape
    maxM = np.max(M)
    Mt = M + 0.0
    indzero = Mt<threshold
    indone = Mt>threshold
    Mt[indzero] = 0
    Mt[indone] = 1

    Mx=nx.from_numpy_matrix(Mt)
        
    all_cliques_iter = nx.enumerate_all_cliques(Mx)
    
    # initialize the Hasse graph (diagram)
    hasse = nx.DiGraph()

    # go through the simplices, create nodes
    all_cliques = []
    all_cliques_fzst = []
    all_cliques_sz = []
    keeps = []
    for clq in all_cliques_iter:
        sclq = sorted(clq)
        v = np.asarray(sclq)
        u = frozenset(str(ui) for ui in sclq)
        sz = v.size
    
        if maxOrder != None and sz>maxOrder:
            break
            # The enumerate_all_cliques iter is ordered by clique size, so we can break to truncate large cliques
        
        if sz == 1:
            prb = 1
        else:
            # Normalize Fischer-z correlations to range < 1, and take product to assign node probabilities
            prb = np.prod(abs(np.asarray([M[a,b] for a,b in itertools.combinations(v,2)])/maxM))
            #prb = np.prod(.5*(np.log(1+prb) - np.log(1-prb)))
            #print([sz, u, prb])
        
        hasse.add_node(u,size=sz,bNodes=v,prob=prb,name=u)
        
        all_cliques.append(v)
        all_cliques_fzst.append(u)
        all_cliques_sz.append(sz)
        
    sz_all_cliques = np.size(all_cliques)        
        
    # create edges in the Hasse graph (diagram)
    # Start from the top and work down
    for i in np.arange(sz_all_cliques-1,-1,-1):
        u = all_cliques_fzst[i]
        v = hasse.nodes[u]['bNodes']
        sz = hasse.nodes[u]['size']
        if sz>1:
            for vv in itertools.combinations(v,sz-1):
                vvv = frozenset(str(vi) for vi in sorted(vv))
                # initially weight directed edges based on target node probabilities
                hasse.add_edge(u, vvv, weight=hasse.nodes[vvv]['prob'])                    
                hasse.add_edge(vvv, u, weight=hasse.nodes[u]['prob'])                        
                    
    for i in range(sz_all_cliques):
        u = all_cliques_fzst[i]
        sz = all_cliques_sz[i]
        hi_sz = sz+1
        probs_hi = 0
        lo_sz = sz-1
        probs_lo = 0
        for nb in hasse.neighbors(u):
            if hasse.nodes[nb]['size']==hi_sz:
                probs_hi += hasse.nodes[nb]['prob']
            elif hasse.nodes[nb]['size']==lo_sz:
                probs_lo += hasse.nodes[nb]['prob']
        for nb in hasse.neighbors(u):
            if hasse.nodes[nb]['size']==hi_sz:
                if probs_lo > 0:
                    hasse.add_edge(u, nb, weight=hasse.nodes[nb]['prob']/probs_hi/2)                    
                else:
                    hasse.add_edge(u, nb, weight=hasse.nodes[nb]['prob']/probs_hi)                    
            elif hasse.nodes[nb]['size']==lo_sz:
                if probs_hi > 0:
                    hasse.add_edge(u, nb, weight=hasse.nodes[nb]['prob']/probs_lo/2)                    
                else:
                    hasse.add_edge(u, nb, weight=hasse.nodes[nb]['prob']/probs_lo)                    
            
    return hasse    
