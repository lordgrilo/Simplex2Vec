import networkx as nx
import itertools
from tqdm import tqdm_notebook

def get_projected_graph(simplex_list, info = False):
    g = nx.Graph()
    for simplex in tqdm_notebook(simplex_list):
        for u,v in itertools.combinations(simplex,2):
            if g.has_edge(u,v):
                g[u][v]["simplices"].add(simplex)
            else:
                g.add_edge(u,v, simplices = set([simplex]))    
    if info:
        print(nx.info(g))
    return g

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

# def get_open_closed_triangles(g, info = True):
#     # Slower implementation through common neighbors
#     closed_triangles = set([])
#     open_triangles = set([])
#     for u,v in tqdm_notebook(list(g.edges())):
#         common_neighbors = nx.common_neighbors(g,u,v)
#         for w in common_neighbors:
#             common_simplices = g[u][w]["simplices"].intersection(g[v][w]["simplices"])
#             if len(common_simplices):
#                 triple_common_simplices = common_simplices.intersection(g[u][v]["simplices"])
#                 if len(triple_common_simplices):
#                     closed_triangles.add(frozenset((u,v,w)))
#                 else:
#                     open_triangles.add(frozenset((u,v,w)))
#             else:
#                 open_triangles.add(frozenset((u,v,w)))
#     if info:
#         print("Open: {}, Closed: {}".format(len(open_triangles), len(closed_triangles)))
#     return open_triangles, closed_triangles