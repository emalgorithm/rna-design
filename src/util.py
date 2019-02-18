import networkx as nx
import matplotlib.pyplot as plt
import os
import pickle
import RNA
import numpy as np


def dotbracket_to_graph(dotbracket):
    G = nx.Graph()
    bases = []

    for i, c in enumerate(dotbracket):
        if c == '(':
            bases.append(i)
        elif c == ')':
            neighbor = bases.pop()
            G.add_edge(i, neighbor, edge_type='base_pair')
        elif c == '.':
            G.add_node(i)
        else:
            print("Input is not in dot-bracket notation!")
            return None

        if i > 0:
            G.add_edge(i, i - 1, edge_type='adjacent')
    return G


def draw_rna_graph(rna_graph):
    edges = list(rna_graph.edges(data=True))
    colors = ['black' if edge[2]['edge_type'] == 'adjacent' else 'blue' for edge in edges]
    nx.draw(rna_graph, edge_color=colors)
    plt.show()


def get_family_to_sequences():
    family_sequences_path = '../data/family_rna_sequences/'
    rna_family_files = sorted(os.listdir(family_sequences_path))
    family_to_sequences = {}

    for file in rna_family_files:
        if 'RF' in file:
            family = file[:7]
            family_sequences = pickle.load(open(family_sequences_path + file, 'rb'))
            family_to_sequences[family] = family_sequences

    return family_to_sequences


def get_sequences_with_folding(family='RF00002'):
    file = '../data/family_rna_sequences/{}.pkl'.format(family)
    sequences = pickle.load(open(file, 'rb'))
    sequences_with_folding = [(sequence, RNA.fold(sequence)[0]) for sequence in sequences]

    return sequences_with_folding


def get_all_sequences():
    family_to_sequences = get_family_to_sequences()
    all_sequences = np.array(list(family_to_sequences.values()))
    all_sequences = [item for sublist in all_sequences for item in sublist]

    return all_sequences
