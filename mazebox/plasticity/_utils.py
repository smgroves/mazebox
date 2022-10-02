import pandas as pd
from graph_tool import all as gt

# import graph_tool as gt
import graph_tool.inference as gt_in
from graph_tool.inference import BlockState
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import linkage, dendrogram
from graph_tool.topology import label_components

# from extra import graph_fit_edits
import seaborn as sns
import os.path as op
import resource
import os
from scipy import stats
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import pickle
import sklearn.model_selection as ms
from numpy.random import poisson
from graph_tool.generation import random_graph

# Given a graph calculate the graph condensation (all nodes are reduced to strongly
# connected components). Returns the condensation graph, a dictionary mapping
# SCC->[nodes in G], as well as the output of graph_tool's label_components.
# I often use this on the output of graph_sim.prune_stg_edges, or a deterministic stg
def condense(G, directed=True, attractors=True):
    # label_components comes from graph_tool directly
    components = label_components(G, directed=directed, attractors=attractors)
    c_G = gt.Graph()
    c_G.add_vertex(n=len(components[1]))

    vertex_dict = dict()
    for v in c_G.vertices():
        vertex_dict[int(v)] = []
    component = components[0]

    for v in G.vertices():
        c = component[v]
        vertex_dict[c].append(v)
        for w in v.out_neighbors():
            cw = component[w]
            if cw == c:
                continue
            if c_G.edge(c, cw) is None:
                edge = c_G.add_edge(c, cw)
    return c_G, vertex_dict, components


def prune_stg_edges(stg, edge_weights, n, threshold=0.5):
    d_stg = gt.Graph()
    for edge in stg.edges():
        if edge_weights[edge] * n > threshold:
            d_stg.add_edge(edge.source(), edge.target())
    print("Finding strongly connected components")
    comps, hists, atts = label_components(d_stg, directed=True, attractors=True)
    return d_stg, comps, hists, atts


def load_network(
    fname, remove_selfloops=False, add_selfloops_to_sources=True, header=None
):
    G = gt.Graph()
    infile = pd.read_csv(fname, header=header, dtype="str")
    vertex_dict = dict()
    vertex_names = G.new_vertex_property("string")
    vertex_source = G.new_vertex_property("bool")
    vertex_sink = G.new_vertex_property("bool")
    for tf in set(list(infile[0]) + list(infile[1])):
        v = G.add_vertex()
        vertex_dict[tf] = v
        vertex_names[v] = tf

    for i in infile.index:
        if not remove_selfloops or infile.loc[i, 0] != infile.loc[i, 1]:
            v1 = vertex_dict[infile.loc[i, 0]]
            v2 = vertex_dict[infile.loc[i, 1]]
            if v2 not in v1.out_neighbors():
                G.add_edge(v1, v2)

    G.vertex_properties["name"] = vertex_names

    for v in G.vertices():
        if v.in_degree() == 0:
            if add_selfloops_to_sources:
                G.add_edge(v, v)
            vertex_source[v] = True
        else:
            vertex_source[v] = False
        if v.out_degree() == 0:
            vertex_sink[v] = True
        else:
            vertex_sink[v] = False

    G.vertex_properties["sink"] = vertex_sink
    G.vertex_properties["source"] = vertex_source

    return G, vertex_dict


def random_graph_generator():

    g = gt.random_graph(100, lambda: (poisson(2), poisson(2)))
    state = gt_in.minimize.minimize_blockmodel_dl(g)

    # Now we run 100 sweeps of the MCMC with zero temperature, as a
    # refinement. This is often not necessary.

    state.multiflip_mcmc_sweep(beta=np.inf, niter=100)

    state.draw(output="example-pp.svg")

    # comp, hist, is_attractor = gt.label_components(g, attractors=True)
    # print(comp.a)
    c_G, vertex_dict, comp = condense(g)
    print(comp[0])
    gt.graph_draw(c_G, output="example_cg.pdf")

    # vertex_colors = g.new_vertex_property("vector<float>")
    # for v in g.vertices():
    #     vertex_colors[v] = vertex_dict[v]
    vprops = {"shape": "circle", "size": 20, "pen_width": 1, "fill_color": comp[0]}

    gt.graph_draw(g, vprops=vprops, output="example_g.pdf")


# random_graph_generator()
#
# g = gt.collection.data["football"]
# state = gt_in.minimize.minimize_blockmodel_dl(g)
# state.draw( output="football-sbm-fit.pdf")


def condense_tm(T, adata, groupby):
    """

    :param T: transition matrix
    :param adata: adata object
    :param groupby: attribute in adata.obs to group cells by
    :return: a new transition matrix that has been condensed
    """
    # for each group of cells, we need to calculate the probability of transitioning to another group of cells (or self transition)
