import networkx as nx
from collections import defaultdict
import itertools 

def make_graph(df):
    companies = defaultdict(set)
    investors = defaultdict(set)
    edges = []
    for _, row in df.iterrows():
        companies[row['Company_Name']].add(row['Investor_Name'])
        investors[row['Investor_Name']].add(row['Company_Name'])
        # add to edges a tuple of (company, investor, {weight: amout_usd})
        edges.append((row['Company_Name'], row['Investor_Name'], {"Weight": row['Raised_Amount_USD']} ))

    attr_dic = {x:y for x, y in zip(df['Company_Name'], df['Company_Market'])}
    attr_dic_fixed = {x:'Investors' for x in df['Investor_Name']}
    attr_dic_fixed.update(attr_dic)
    companies_edges.append(companies)
    investors_edges.append(investors)
    all_edges.append(edges)
    graph = nx.Graph()
    graph.add_edges_from(all_edges)
    nx.set_node_attributes(graph, attr_dic_fixed, name="Sector")
    return graph


def make_companies_graph(df):
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