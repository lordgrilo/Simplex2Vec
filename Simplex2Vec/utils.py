#! usr/bin/python3

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import warnings
import json
import itertools
from tqdm import tqdm

def read_simplex_json(path):

    with open(path, 'r') as fh:
        list_simplices = json.load(fh)


    list_simplices=[frozenset([str(i) for i in li]) for li in list_simplices]

    return list_simplices
