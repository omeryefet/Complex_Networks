def Communitys(graph):
   
    #Using the Kernighan–Lin algorithm
    KL_communities_generator = community.kernighan_lin_bisection(graph, max_iter=10)
    #Measuring partitions by modularity
    print (community.modularity(graph, KL_communities_generator))

    #Communities with modularity
    greedy_mod=greedy_modularity_communities(graph)
    print (community.modularity(graph, greedy_mod))
    
    #Giravn _newman communities:
    GN_communities_generator = c.girvan_newman(graph)
    for i in range(10):
        communities= next(GN_communities_generator)

    GN_comm_sets = sorted(map(sorted, communities))
    #Measuring partitions by modularity
    print (community.modularity(graph, GN_comm_sets))
    
    #Louvain communities:
    partition = community_louvain.best_partition(graph)
    
    d_for_getting_num_of_com={}
    for x in partition:
        if partition[x] not in d_for_getting_num_of_com:
            d_for_getting_num_of_com[partition[x]]=1
    empty_lists = [ [] for l in range(len(d_for_getting_num_of_com) ) ]
    
    for node in partition:
        empty_lists[partition[node]].append(node)
    print (community.modularity(graph,empty_lists))

    return KL_communities_generator, greedy_mod, GN_comm_sets, partition