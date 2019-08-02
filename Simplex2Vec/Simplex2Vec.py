#! usr/bin/python3
import warnings
import os

import networkx as nx
import numpy as np
import pickle as pkl
import random
from platform import python_version

from multiprocessing import Pool
from tqdm import tqdm_notebook
from gensim.models import Word2Vec
from copy import deepcopy

from joblib import Parallel, delayed

from Simplex2Vec_v2.simplex2hasse import *

class Simplex2Vec():

    def __init__(self, G, n_walks=10, walk_length=15, p=10000000000., q=1, hasse_max_order=None, is_quiet = False,
        hasse_weight_scheme='uniform', rw_bias = "LOexp", workers=1, from_hasse=False, from_graph=False):
        '''
        Finds simplicial complexes in a network, creates Hasse diagram with chosen weighting scheme and performs the random walks.
        G : NetworkX Graph
            Graph to embed if from_hasse=False. To load Hasse diagram use the from_hasse_diagram method instead.
        n_walks : int (dafault: 10)
            Number of RWs starting from each node of the Hasse diagram.
        walk_length : int (default: 15)
            Number of nodes visited by each RW.
        p : float (default: 1)
            Return parameter for the RWs (same as in Node2Vec). 
        q : float (default: 1)
            In-Out parameter for the RWs (same as in Node2Vec).
        hasse_max_order : int (default: None)
            Simplex order up to which Hasse diagram must be constructed (or navigated, in case Hasse is externally loaded). Go to maximal order found in data if set to None.
        hasse_weight_scheme : str (default: 'LOa')
            Weighting scheme for construction of the Hasse diagram. Not available if Hasse is externally loaded.
            Must be one of:
            'uniform' : set "weight = 1" to all nodes in the diagram
            'counts' : set "weight = number of encounters of a simplex in the data" to all simplices that are actually found in the data, and weight = 1 to all the others
            'LOlin' : weights heavily biased towards lower-order simplices
            'LOexp' : weights mildly biased towards lower-order simplices
            'HOlin' : weights linearly biased towards higher-order simplices
            'HOexp' : weights hardly biased towards higher-order simplices
        rw_bias : str (default: "LOexp")
            Bias of the random walker to go to the simplices of larger or smaller order. This is extra bias in addition to a weighting scheme. Probability of going to a node u is calculated as bias_coefficient(u)*weight(u)*p_return(u). For more information on weighting schemes and biases please refer to the code.
            Possible values:
            'LOlin' : random walk is mildly biased to go to lower-order simplices (coefficient 1/(len(simplex)-1))
            'LOexp' : random walk is heavy biased to go to lower-order simplices (coefficient 1/2**(len(simplex)-1)) 
            'HOlin' : random walk is mildly biased to go to higher-order simplices (coefficient len(simplex)-1)
            'HOexp' : random walk is heavily biased to go to higher-order simplices (coefficient 2**(len(simplex)-1)) 
            Remark: any other string entry will result in an unbiased random walk 
        workers : int (default: 1)
            Number of processes to be used.
        from_hasse : bool (default: False)
            Not meant to be set by the user, keep the default value.
        from_graph : bool (default: False)
            Not meant to be set by the user, keep the default value.
        is_quiet : bool (default: False)
            Show the tqdm progress bar when computing the random walks or leave it as it is
        '''
        # constants for probability calculation 
        self.PROBABILITIES_KEY = "probabilities"
        self.FIRST_TRAVEL_KEY = "first_travel_probabilities"
        self.NEIGHBORS_KEY = "neighbors"
        
        # get Python version 
        pytv = python_version().split(".")[:2]
        pytv = ".".join(pytv)
        self.python_ver = float(pytv)

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
        
        # Check type of is_quiet and record the value
        if isinstance(is_quiet, bool):
            self.is_quiet = is_quiet
        else:
            raise ValueError('is_quiet must be a bool.')
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

        self._precompute_probabilities(bias = rw_bias)
        self._hasse_RW()
        
        return

		

    @classmethod
    

    def from_hasse_diagram(cls, G_hasse, n_walks=1, walk_length=10, p=1, q=1, rw_bias = "nobias", is_quiet = False, hasse_max_order=None, workers=1):

        g = deepcopy(G_hasse)

        return cls(g, n_walks, walk_length, p, q, hasse_max_order, workers=workers, 
                   from_hasse=True, rw_bias = rw_bias, is_quiet = is_quiet)


    @classmethod
    def read_hasse_diagram(cls, filename, n_walks=1, walk_length=10, p=1, q=1, hasse_max_order=None, workers=1):

        if os.path.exists(filename):
            with open(filename, 'rb') as fh:
                g = pkl.load(fh)
        else:
            raise FileNotFoundError('Could not find {}')
                
        return cls(g, n_walks, walk_length, p, q, hasse_max_order, hasse_weight_scheme=None,  workers=workers, from_hasse=True)

    @classmethod
    def from_hasse_DiGraph(cls, G_hasse, n_walks=10, walk_length=10, p=1, q=1, hasse_max_order=None, workers=1):

        g = deepcopy(G_hasse)

        return cls(g, n_walks, walk_length, p, q, hasse_max_order, hasse_weight_scheme=None,  workers=workers, from_hasse=True)
    
    @classmethod
    def from_graph(cls, inData, threshold=0.5, n_walks=10, walk_length=5, p=1, q=1, hasse_max_order=None, workers=1):

        g = graph2hasse_proportional(inData, threshold, hasse_max_order)

        return cls(g, n_walks=n_walks, walk_length=walk_length, p=p, q=q, workers=workers, hasse_max_order=hasse_max_order, from_graph=True)
    

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

        elif 'LOlin' == w_scheme:
            self.hasse = simplex2hasse_LOlinear(cliques, self.max_order)

        elif 'LOexp' == w_scheme:
            self.hasse = simplex2hasse_LOexponential(cliques, self.max_order)

        elif 'HOlin' == w_scheme:
            self.hasse = simplex2hasse_HOlinear(cliques, self.max_order)
            
        elif 'HOexp' == w_scheme:
            self.hasse = simplex2hasse_HOexponential(cliques, self.max_order)
            
        elif 'proportional' == w_scheme:
            self.hasse = simplex2hasse_proportional( cliques , self.max_order)
        else:
            raise ValueError('{} is not a valid weighting scheme.'.format(w_scheme))

        return

    def _precompute_probabilities(self, bias = "LOexp"):

        first_travel_probababilities_computed = set()
        
        def __bias_coefficient():
            if bias == 'LOexp':
                if len(destination) > 1:
                    return 1/2**(len(destination)-1)
                else:
                    return 1
            elif bias == "LOlin":
                if len(destination) > 1:
                    return 1/(len(destination)-1)
                else:
                    return 1
            elif bias == 'HOlin':
                if len(destination) > 1:
                    return (len(destination)-1)
                else:
                    return 1
            elif bias == "HOexp":
                if len(destination) > 1:
                    return 2**(len(destination)-1)
                else:
                    return 1
            else:
                return 1

        for source in tqdm_notebook(self.hasse.nodes(), desc = "Precomputing probabilities"):
            # Init probabilities dict 
            if self.PROBABILITIES_KEY not in self.hasse.node[source]:
                self.hasse.node[source][self.PROBABILITIES_KEY] = dict()
                
            for current_node in self.hasse.neighbors(source):
                # Init probabilities dict 
                if self.PROBABILITIES_KEY not in self.hasse.node[current_node]:
                    self.hasse.node[current_node][self.PROBABILITIES_KEY] = dict()
                    
                with_return_weights = []
                first_travel_weights = []
                if self.NEIGHBORS_KEY in self.hasse[current_node]:
                    current_neighbors = self.hasse.node[current_node][self.NEIGHBORS_KEY]
                else:
                    current_neighbors = list(self.hasse.neighbors(current_node))
                current_weight = [self.hasse.nodes[n]['weight'] for n in current_neighbors]
                
                # Calculate unnormalized weights
                for destination in current_neighbors:
                    first_travel_weights.append(1)
                    if destination == source:
                        with_return_weights.append(__bias_coefficient()/self.p)
                    elif self.hasse.has_edge(source, destination):
                        with_return_weights.append(__bias_coefficient())
                    else:
                        with_return_weights.append(__bias_coefficient()/self.q)
                
                # Compute total with return probabilities
                with_return_weights = np.asarray(with_return_weights)
                prob = np.multiply(with_return_weights, current_weight)
                prob = prob/np.sum(prob)
                self.hasse.node[current_node][self.PROBABILITIES_KEY][source] = prob
                
                # Compute first visit probabilities
                if current_node not in first_travel_probababilities_computed:
                    first_travel_weights = np.asarray(first_travel_weights)
                    prob = np.multiply(first_travel_weights, current_weight)
                    prob = prob/np.sum(prob)
                    self.hasse.node[current_node][self.FIRST_TRAVEL_KEY] = prob
                    first_travel_probababilities_computed.add(current_node)
                self.hasse.node[current_node][self.NEIGHBORS_KEY] = current_neighbors


    def _hasse_RW_helper(self, input_values):
        cpu_index, node_list = input_values

        def _RW_iteration(prev_node, current_node, rest_steps):
            if rest_steps > 0:
                neighbors_list = self.hasse.node[current_node][self.NEIGHBORS_KEY]
                if None == prev_node:
                    prob = self.hasse.node[current_node][self.FIRST_TRAVEL_KEY]
                else:
                    prob = self.hasse.node[current_node][self.PROBABILITIES_KEY][prev_node]
                if self.python_ver > 3.5:
                    # faster implementation of choice from list
                    next_node = random.choices(neighbors_list, weights=prob)[0]
                else:
                    next_node = np.random.choice(neighbors_list, p=prob)
                _RW_iteration(current_node, next_node, rest_steps-1)
                walk.append(next_node)

        walk_list = []
        # This line is the strange hack
        print(' ', end='', flush=True)
        if self.is_quiet:
            print("Generating walks : CPU: {}".format(cpu_index))
            iterator = node_list
        else:
            iterator = tqdm_notebook(node_list, 
                                          desc = "Generating walks : CPU: {}".format(cpu_index), 
                                          position = cpu_index)
        for current_node in iterator:
            walk_counter = 0
            while walk_counter < self.n_walks:
                walk = [current_node]
                if 0 != len(list(self.hasse.neighbors(current_node))):
                    prev_node = None
                    rest_steps = self.walk_length-1
                    _RW_iteration(prev_node, current_node, rest_steps)
                else:
                    walk = walk*self.walk_length
                walk_list.append(walk)
                walk_counter += 1

        return walk_list

    
    def _hasse_RW(self):
        
        self.walks = []
        nodes = np.asarray([n for n in self.hasse.nodes()])
        nodes_split = np.array_split(nodes, self.workers)
        if self.workers != 1:
            with Parallel(n_jobs=self.workers, backend = "multiprocessing") as parallel:
                self.walks = parallel(delayed(self._hasse_RW_helper)(x) for x in enumerate(nodes_split))
                
        # old version of parallel execution, change if you have problems with joblib
#             mp_pool = Pool(processes = self.workers)
#             self.walks.extend(mp_pool.map(self._hasse_RW_helper, enumerate(nodes_split)))
#             mp_pool.close()
            
        else:
            self.walks.extend(list(map(lambda x: _hasse_RW_helper(x), enumerate(nodes_split))))
        
        flatten = lambda l: [item for sublist in l for item in sublist]
        flatten_walks = flatten(self.walks)

        l_walks = []
        for rw in flatten_walks:
            rw_parsed = []
            for u in rw:
                u_int = sorted([int(x) for x in list(u)])
                rw_parsed.append("-".join([str(x) for x in u_int]))
            l_walks.append(rw_parsed)
        self.walks = l_walks
        del l_walks
        print("DONE!")
        return


    def fit(self, **skip_gram_params):

        if 'workers' not in skip_gram_params:
            skip_gram_params['workers'] = self.workers
        print("Fitting the Word2Vec..")
        self.model = Word2Vec(self.walks, compute_loss=True, **skip_gram_params)

        return self.model 
    
    def refit(self, epochs=1):
        self._hasse_RW()
        self.model.train(self.walks, compute_loss=True, total_examples = len(self.walks), epochs = epochs)
        return self.model