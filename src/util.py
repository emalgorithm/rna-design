import networkx as nx
import matplotlib.pyplot as plt


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
            pass
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
