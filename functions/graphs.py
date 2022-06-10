import networkx as nx
from collections import defaultdict
import itertools 

def make_companies_graph(df):
    """
    Create the companies projection graph
    We say that 2 companies are connected, if they share atleast one common investor
    :param df: pandas.Dataframe
    :return: nx.Graph
    """
    investor_names = df['Investor_Name'].unique()
    edges = []
    for company in investor_names:
        invested_in_companies = df.loc[df['Investor_Name'] == company]['Company_Name'].unique() # Get all companies invested by an investor
        if len(invested_in_companies) >=2:
            combinations =  itertools.combinations_with_replacement(invested_in_companies, 2)
            for c in combinations:
                if c[0] != c[1]:
                    edges.append(c)
    attr_dic = {x:y for x, y in zip(df['Company_Name'], df['Company_Market'])}
    graph = nx.Graph()
    graph.add_edges_from(edges)

    nx.set_node_attributes(graph, attr_dic, name="Sector")
    return graph

def make_investors_graph(df):
    """
    Create the companies projection graph
    We say that 2 investors are connected, if they share atleast one common company they invested in
    :param df: _description_
    :return: _description_
    """
    company_names = df['Company_Name'].unique()
    edges = []
    for company in company_names:
        investors = df.loc[df['Company_Name'] == company]['Investor_Name'].unique() # Get all unique investors for a company
        if len(investors) >=2:
            combinations =  itertools.combinations_with_replacement(investors, 2)  # Create edges between 2 investors that invested in the same company
            for c in combinations:
                if c[0] != c[1]:
                    edges.append(c)
    graph = nx.Graph()
    graph.add_edges_from(edges)
    return graph