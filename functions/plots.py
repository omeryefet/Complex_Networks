import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import networkx as nx
from collections import defaultdict
import powerlaw

years_range = [(1990, 1994), (1995, 1999), (2000, 2004), (2005,2009), (2010, 2014)]
years_dict = {i:year_range for i,year_range in enumerate(years_range)}

def fit_func(x,a,mu):
    return (a*x)**mu

def plot_knn(graph):
    knn_dict = nx.k_nearest_neighbors(graph)
    k_lst = sorted(knn_dict.keys())
    knn_lst = []
    for k in k_lst:
        knn_lst.append(knn_dict[k]) 
    fig = plt.figure(figsize=(6,6))
    axes = fig.add_axes([1,1,1,1])
    axes.loglog(k_lst,knn_lst,'b.', markersize=15, alpha=0.5)
    axes.set_xlabel('k')
    axes.set_ylabel('knn (k)')
    axes.set_title('Average next neighbor degree')
    try:
        popt, _ = curve_fit(fit_func, np.array(k_lst), np.array(knn_lst),maxfev=5000)
        axes.loglog(np.array(k_lst), fit_func(np.array(k_lst), *popt), '--', c='gray')
        axes.plot(np.array(k_lst), np.array([2*graph.number_of_edges()/graph.number_of_nodes()]*len(k_lst)),label='Random Prediction')
        if popt[1] > 0:
            print(f'Assortative graph with mu {round(popt[1],3)}')
        elif popt[1] < 0:
            print(f'Disassortative graph with mu {round(popt[1],3)}')
        else:
            print(f'Neutral graph (mu = 0)')
    except:
        pass
    # fig.savefig('plots/knn.png', bbox_inches='tight')
    plt.legend()
    plt.show()


def plot_snn(graph):
    knn_dict = nx.k_nearest_neighbors(graph,weight = 'Raised_Amount_USD')
    k_lst = sorted(knn_dict.keys())
    snn_lst = []
    for k in k_lst:
        snn_lst.append(knn_dict[k]) 
    fig = plt.figure(figsize=(6,6))
    axes = fig.add_axes([1,1,1,1])
    axes.loglog(k_lst,snn_lst,'b.', markersize=15, alpha=0.5)
    axes.set_xlabel('k')
    axes.set_ylabel('snn (k)')
    axes.set_title('Average next neighbor degree')
    try:
        popt, _ = curve_fit(fit_func, np.array(k_lst), np.array(snn_lst),maxfev=5000)
        print(popt)
        axes.loglog(np.array(k_lst), fit_func(np.array(k_lst), *popt), '--', c='gray')
        axes.plot(np.array(k_lst), np.array([2*graph.number_of_edges()/graph.number_of_nodes()]*len(k_lst)),label='Random Prediction')
        if popt[1] > 0:
            print(f'Assortative graph with mu {round(popt[1],3)}')
        elif popt[1] < 0:
            print(f'Disassortative graph with mu {round(popt[1],3)}')
        else:
            print(f'Neutral graph (mu = 0)')
    except:
        pass
    # fig.savefig('plots/knn.png', bbox_inches='tight')
    plt.legend()
    plt.show()


def plot_distance_dist(graph):
    shortest_path = nx.shortest_path_length(graph)
    distance_dict = defaultdict(int)
    for i in shortest_path:
        for j in i[1].values():
            if j > 0:
                distance_dict[j] += 1
    distances = []
    counts = []
    for item in distance_dict.items():
        distances.append(item[0])
        counts.append(item[1]/np.sum(list(distance_dict.values())))

    plt.figure(figsize=(15,8))
    plt.xlabel("d")
    plt.ylabel("P(d)")
    plt.title('Distribution of distances')
    plt.plot(distances,counts,c='r')

    avg_d=0
    for i in range(len(distances)):
        avg_d+=distances[i]*counts[i]/sum(counts)
    print(f'Average distance is {round(avg_d,3)}')
    print(f'Maximum distance is {max(distances)}')

def plot_degree_distribution(graph):
    # nodes = pd.DataFrame(list(graph.nodes))
    print(nx.info(graph))
    # degrees = [graph.degree(n) for n in graph.nodes()]
    degree_dict = {'In':graph.in_degree,'Out':graph.out_degree,'Total':graph.degree}
    dic = defaultdict() # key=in/out/total : value=count_deg
    for item in degree_dict.items():
        count_deg = defaultdict(int)
        for _, degree in item[1]:
            count_deg[degree] += 1 / len(item[1])
            count_deg[degree] += 1
        dic[item[0]] = count_deg

    for j,dict in dic.items():
        fig, axs = plt.subplots(1,2,figsize=(20,5))
        for i in range(2):
            axs[i].scatter(dict.keys(), dict.values())
            # mean_k = np.mean([item[0]*item[1] for item in dict.items()])
            # max_k = max(dict.keys())
            # a = np.linspace(1, max_k, len(list(dict.keys())))
            # axs[i].plot(dict.keys(), poisson.pmf(a, mu=5))
            # axs[i].plot(np.linspace(0, max_k, 100), np.exp(np.linspace(0, max_k, 100)))
            # axs[i].plot(np.log(np.linspace(1, max_k*10, 1000000)), np.log(np.exp(np.linspace(1, max_k*10, 1000000))))
            axs[i].set_xlabel("k")
            axs[i].set_ylabel("P(k)")
            title = f'{j}-Degree Distribution {years_range[4][0]}-{years_range[4][1]}'
            if i>0:
                axs[i].set_yscale('log')
                axs[i].set_xscale('log')
                # mean_k = np.mean([item[0]*item[1] for item in dict.items()])
                # max_k = max(dict.keys())
                # a = np.linspace(1, max_k, max_k)
                # from math import e
                # import math
                # b = [((e**(-mean_k))*(mean_k**i))/(math.factorial(i)) for i in a]
                # axs[i].plot(a,b)
                # from scipy.stats import poisson
                # axs[i].plot(np.linspace(1, max_k, 1000000), poisson.pmf((np.linspace(1, max_k, 1000000)),mu=mean_k))
                title += ' (log scale)'
            axs[i].set_title(title)


def plot_gamma(graph):
    
    degree_dict = {'In': graph.in_degree,'Out': graph.out_degree,'Total': graph.degree}
    degree_sequence = sorted([d for _, d in degree_dict['Total']],reverse=True)
    figPDF = powerlaw.plot_pdf(degree_sequence,color='b')
    figPDF.set_ylabel(r"P(k)")
    figPDF.set_xlabel(r"k")
    fit = powerlaw.Fit(degree_sequence,discrete=True)
    gamma = fit.power_law.alpha
    print(f"Gamma (Total) = {gamma}")

def plot_giant_component(graph, num):
    if nx.is_connected(graph):
      print("Diameter(G)=",nx.diameter(graph))
    else:
        print(f'Graph of years {years_dict[num][0]}-{years_dict[num][1]} is unconnected')
        print("Size giant componenet =", len(max(nx.connected_components(graph),key=len)))

def plot_weight_dist(graph):
    
    weight_list = sorted(list((nx.get_edge_attributes(graph,'Raised_Amount_USD').values())))
    weight_dict = defaultdict(int)
    for w in weight_list:
        weight_dict[w] += 1
    weights = []
    counts = []
    for weight,cnt in weight_dict.items():
        weights.append(weight)
        counts.append(cnt)
    plt.figure(figsize=(15,8))
    plt.xlabel("Weights (Hundred M)")
    plt.ylabel("Number of weights")
    plt.title(f'Distribution of weights')
    plt.plot(weights,counts,c='r')
    
    avg_w=0
    for i in range(len(weights)):
        avg_w+=weights[i]*counts[i]/sum(counts)
    print(f' The average weight is {round(avg_w,3)} USD')

def plot_company_market_distribution(df): 
    df.Company_Market.value_counts().head(15).plot.pie(figsize=(15,10),autopct='%1.0f%%',fontsize=13)
    plt.title('Investments Distribution by Company Market\n\n', fontsize=20, fontweight='bold',color='tomato')
    plt.axis('equal')

def plot_clustering_coefficient(graph, num):
    # num is used only to save the image with number
    clustering_dict = defaultdict(list)
    for node in graph.nodes():
        k = graph.degree(node)
        clustering_dict[k].append(nx.clustering(graph,node))
    k_list = sorted(clustering_dict.keys())
    clustering_list = []
    for k in k_list:
        clustering_list.append(np.mean(clustering_dict[k])) 
    fig = plt.figure(figsize=(6,6))
    axes = fig.add_axes([1,1,1,1])
    axes.loglog(k_list,clustering_list,'b.', markersize=15, alpha=0.5)
    axes.set_xlabel('k')
    axes.set_ylabel('C(k)')
    axes.set_title('Average Clustering Coefficient (log scale)')
    # fig.savefig(f'plots/clustering_coefficient_{num}.png', bbox_inches='tight')
    N = graph.number_of_nodes()
    L = graph.number_of_edges()
    p = (2*L)/(N*(N-1))
    print('Average clustering coefficient for random graph is {}'.format(p))
    axes.plot(k_list,[p]*len(k_list))
    plt.show()



def plot_histogram(hist,title):
    plt.rc('font', size=10)  # controls default text sizes
    plt.title(title)
    plt.bar(hist.keys(), hist.values())
    plt.xticks(rotation=45)
    plt.show()


