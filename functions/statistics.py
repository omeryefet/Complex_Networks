import networkx as nx
from collections import defaultdict
import pandas as pd

def print_degree_correlation_coefficient(graph):
    calc = nx.degree_pearson_correlation_coefficient(graph,x='in', y='in', weight='Raised_Amount_USD', nodes=None)
    if calc > 0:
        print(f'Assortative graph with r = {round(calc,3)}')
    elif calc < 0:
        print(f'Disassortative graph with r = {round(calc,3)}')
    else:
        print(f'Neutral graph (r = 0)')

def get_in_out_nodes(g, investors:pd.DataFrame, company_name: pd.DataFrame):
    """
    Split degrees to companies and investors (inbound, outbound)
    :param g: graph
    :param investors: investors DF
    :param company_name: comapnies DF
    """
    all_degrees = g.degree
    in_nodes = defaultdict(int)
    out_nodes = defaultdict(int)

    for company, n_edges in all_degrees:
        if company in investors.values:
            out_nodes[company] = n_edges
        elif company in company_name.values:
            in_nodes[company] = n_edges

    print("Invested in companies (inbound)")
    # print(dict(in_nodes))
    print(in_nodes)
    print("\n")
    print("Investing companies (outbound)")
    # print(dict(out_nodes))
    print(out_nodes)
    return in_nodes, out_nodes

def avg_degree(nodes:dict):
    """
    Calculate the average degree of the nodes in the graph
    :param nodes: dictionary of node_name(str): degree(int)
    """
    return sum([degree for degree in nodes.values()]) / len(nodes)
