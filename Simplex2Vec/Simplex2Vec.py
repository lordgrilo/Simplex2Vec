#! usr/bin/python3
import warnings
import os

import networkx as nx
import numpy as np
import pickle as pkl

from multiprocessing import Pool
from tqdm import tqdm
from gensim.models import Word2Vec
from copy import deepcopy

from Simplex2Vec.simplex2hasse import *


class Simplex2Vec():

    def __init__(self, G, n_walks=10, walk_length=15, p=1, q=1, hasse_max_order=None, hasse_weight_scheme='LOa',  workers=1, from_hasse=False, from_graph=False):
        '''
        Finds simplicial complexes in a network, creates Hasse diagram with chosen weighting scheme and performs the random walks.
        G : NetworkX Graph
            Graph to embed if from_hasse=False. To load Hasse diagram use the from_hasse_diagram method instead.
        n_walks : int (dafault: 10)
            Number of RWs starting from each node of the Hasse diagram.
        walk_length : int (default: 15)
            Number of nodes visited by each RW.
        p : float (default: 1)
            Return parameter for the RWs (same as in Node2Vec). Set it to a very high value to avoid backtracking.
        q : float (default: 1)
            In-Out parameter for the RWs (same as in Node2Vec).
        hasse_max_order : int (default: None)
            Simplex order up to which Hasse diagram must be constructed (or navigated, in case Hasse is externally loaded). Go to maximal order found in data if set to None.
        hasse_weight_scheme : str (default: 'LOa')
            Weighting scheme for construction of the Hasse diagram. Not available if Hasse is externally loaded.
            Must be one of:
            'uniform' : set weight=1 to all nodes in the diagram
            'counts' : set weight=1 to all simplices that are actually found in the data, weight=0 to all the others
            'LO' : weights heavily biased towards lower-order simplices
            'LOa' : weights mildly biased towards lower-order simplices
        workers : int (default: 1)
            Number of processes to be used.
        from_hasse : bool (default: False)
            Not meant to be set by the user, keep the default value.
        from_graph : bool (default: False)
            Not meant to be set by the user, keep the default value.
        '''

        #Check and set G
        if not isinstance(G, nx.Graph):
            raise ValueError('G must be a NetworkX Graph.')

        #Check and set p
        if p > 0:
            self.p = p
        else:
            raise ValueError('p must be a positive number.')

        #Check and set q
        if q > 0:
            self.q = q
        else:
            raise ValueError('q must be a positive number.')

        #Check and set max hasse diagram order
        if None == hasse_max_order:
            self.max_order = hasse_max_order
        elif int == type(hasse_max_order) and hasse_max_order > 0:
            self.max_order = hasse_max_order
        else:
            raise ValueError('hasse_max_order must be a positive integer.')

        #Check and set number of walks per node
        #if None == n_walks:
        #    self.n_walks = 10
        if int == type(n_walks) and n_walks > 0:
            self.n_walks = n_walks
        else:
            raise ValueError('n_walks must be a positive integer.')

        #Check and set number of parallel workers
        if int == type(workers) and (workers > 0 or -1 == workers):
            self.workers = workers
        else:
            raise ValueError('workers must be a positive integer or -1 (to use all available threads).')


        #load external Hasse diagram
        if from_hasse:

            if None != self.max_order:

                l_remove = [n for n in G.nodes() if len(n)>self.max_order+1]

                if 0 == len(l_remove):

                    max_ord = len(max(G.nodes(), key=lambda x: len(x)))-1
                    message = 'No simplices of order bigger than {} (MAX ORDER = {}). Maybe need to re-build Hasse diagram?'.format(self.max_order, max_ord)
                    warnings.warn(message, UserWarning)
                else:
                    G.remove_nodes_from(l_remove)

            self.hasse = G

        elif from_graph:
            
            self.hasse = G
            
        else:
        #build hasse diagram

            cliques = list(nx.find_cliques(G))
            cliques = [frozenset([str(i) for i in li]) for li in cliques]

            self._build_hasse(hasse_weight_scheme, cliques)


        #Check and set walk length
        if None == walk_length:
            self.walk_length = self.hasse.number_of_nodes()
        elif int == type(walk_length) and walk_length > 0:
            self.walk_length = walk_length
        else:
            raise ValueError('walk_length must be a positive integer.')

        # Preserve flags
        self.from_graph = from_graph
        self.from_hasse = from_hasse
        
        self._hasse_RW()

        return

		

    @classmethod
    def from_hasse_diagram(cls, G_hasse, n_walks=10, walk_length=10, p=1, q=1, hasse_max_order=None, workers=1):

        g = deepcopy(G_hasse)

        return cls(g, n_walks, walk_length, p, q, hasse_max_order, hasse_weight_scheme=None,  workers=workers, from_hasse=True)

    @classmethod
    def from_hasse_DiGraph(cls, G_hasse, n_walks=10, walk_length=10, p=1, q=1, hasse_max_order=None, workers=1):

        g = deepcopy(G_hasse)

        return cls(g, n_walks, walk_length, p, q, hasse_max_order, hasse_weight_scheme=None,  workers=workers, from_hasse=True)
    
    @classmethod
    def from_graph(cls, inData, threshold=0.5, n_walks=10, walk_length=5, p=1, q=1, hasse_max_order=None, workers=1):

        g = graph2hasse_proportional(inData, threshold, hasse_max_order)

        return cls(g, n_walks=n_walks, walk_length=walk_length, p=p, q=q, workers=workers, hasse_max_order=hasse_max_order, from_graph=True)
    
    @classmethod
    def read_hasse_diagram(cls, filename, n_walks=10, walk_length=10, p=1, q=1, hasse_max_order=None, workers=1):

        if os.path.exists(filename):
            with open(filename, 'rb') as fh:
                g = pkl.load(fh)
        else:
            raise FileNotFoundError('Could not find {}')
                
        return cls(g, n_walks, walk_length, p, q, hasse_max_order, hasse_weight_scheme=None,  workers=workers, from_hasse=True)


    def get_hasse_diagram(self, copy=True):

        if copy:
            return deepcopy(self.hasse)
        else:
            return self.hasse

    def dump_hasse_diagram(self, filename):

        with open(filename, 'wb') as fh:
            pkl.dump(self.hasse, fh, protocol=pkl.HIGHEST_PROTOCOL)

        return



    def _build_hasse(self, w_scheme, cliques):

        if 'uniform' == w_scheme:
            self.hasse = simplex2hasse_uniform(cliques, self.max_order)

        elif 'counts' == w_scheme:
            self.hasse = simplex2hasse_counts(cliques, self.max_order)

        elif 'LO' == w_scheme:
            self.hasse = simplex2hasse_LO(cliques, self.max_order)

        elif 'LOa' == w_scheme:
            self.hasse = simplex2hasse_LOadjusted(cliques, self.max_order)

        elif 'LOl' == w_scheme:
            self.hasse = simplex2hasse_LOlinear(cliques, self.max_order)
            
        elif 'proportional' == w_scheme:
            self.hasse = simplex2hasse_proportional( cliques , self.max_order)

        else:
            raise ValueError('{} is not a valid weighting scheme.'.format(w_scheme))

        return


    def _hasse_RW_helper(self, node):

        walk = [node]
        
        if 0 != len(list(self.hasse.neighbors(node))):

            prev_node = None
            pEqualsq = self.p==self.q
            isDiGraph = isinstance(self.hasse,nx.DiGraph)                        

            for j in range(self.walk_length-1):

                neighb = list(self.hasse.neighbors(node))                
                if isDiGraph:
                    w = [self.hasse.edges[node,n]['weight'] for n in neighb]
                else:
                    w = [self.hasse.nodes[n]['weight'] for n in neighb]

                if None == prev_node or pEqualsq:
                    prob = np.array(w)/np.sum(w)
                else:
                    alpha = []
                    for n in neighb:
                        if n == prev_node:
                            alpha.append(1/self.p)
                        elif self.hasse.has_edge(n, prev_node):
                            alpha.append(1)
                        else:
                            alpha.append(1/self.q)

                    prob = np.multiply(w,alpha)
                    prob = prob/np.sum(prob)

                prev_node = node
                node = np.random.choice(neighb, p=prob)
                walk.append(node)
        else:
            walk = walk*self.walk_length

        return walk

    
    def _hasse_RW(self):

        self.walks = []
        
        nodes = tqdm([n for n in self.hasse.nodes()]*self.n_walks, 'Creating random walks')

        if self.workers!=1:
            p = Pool(processes=self.workers)
            self.walks.extend(p.map(self._hasse_RW_helper, nodes))
            p.close()
        else:
            self.walks.extend(list(map(lambda x: self._hasse_RW_helper(x), nodes)))

        l_walks = []
        for rw in self.walks:
            
            l_walks.append([",".join(sorted(list(u))) for u in rw])

        self.walks = l_walks

        del l_walks

        return


    def fit(self, **skip_gram_params):

        if 'workers' not in skip_gram_params:
            skip_gram_params['workers'] = self.workers
            
        self.model = Word2Vec(self.walks, compute_loss=True, **skip_gram_params)
        
        return self.model
    
    def refit(self, epochs=1):

        self._hasse_RW()
        
        self.model.train(self.walks, compute_loss=True, total_examples = len(s2v.walks), epochs = epochs)

        return self.model
