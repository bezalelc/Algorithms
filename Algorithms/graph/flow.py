from Algorithms.structure.graph_struct import Graph, Vertex
import math


def dfs(G, s, t, theta=None):
    """
    find adding way with dfs

    @params:
        @G: graph
        @s: vertex to start bfs
        @t: vertex to end bfs
        @theta: min flow (for scaling)

    return: dict that contains the edges for adding way from s to t OR False if there is not adding way

    efficiency: O(V+E)
    """
    pi_edges, color = {v: None for v in G.V}, {v: 'white' for v in G.V}

    def dfs_visit(v):
        color[v] = 'gray'
        for e in v.edges:
            if color[e.to] == 'white':
                if theta and e.weight < theta:
                    continue
                pi_edges[e.to] = e
                if e.to == t:
                    return
                else:
                    dfs_visit(e.to)

    dfs_visit(s)
    return pi_edges if pi_edges[t] else False


def bfs(G, s, t, theta=None):
    """
    find adding way with bfs

    @params:
        @G: graph
        @s: vertex to start bfs
        @t: vertex to end bfs
        @theta: min flow (for scaling)

    return: dict that contains the edges for adding way from s to t OR False if there is not adding way

    efficiency: O(V+E)
    """
    pi_edges, color, Q = {v: None for v in G.V}, {v: 'white' for v in G.V}, [s]

    while len(Q):
        v = Q.pop(0)
        for e in v.edges:
            if color[e.to] == 'white':
                if theta and e.weight < theta:
                    continue
                pi_edges[e.to], color[e.to] = e, 'gray'
                if e.to == t:
                    return pi_edges
                Q.append(e.to)

    return False


def ford_fulkerson(G, s, t, func=dfs):
    """
    ford fulkerson algorithm:
        found max flow in the graph from s to t
          if the weight are irrational numbers the algorithm is not sure the algorithm will converge
          if the weight are rational numbers we can multiple all weight with common factor

    @params:
        @G: graph
        @s: vertex to start
        @t: vertex to end
        @func: function to use for find adding way by default is dfs, can be bfs

    return: dict that contain the flow for every edge

    efficiency:
        if func is dfs: O(|E||f*|) when |f*| is max flow
        if func is bfs: O(|V||E|^2)
    """
    G_r, dic = init_G_r(G)
    s, t, way = dic[s], dic[t], True
    while way:
        way = func(G_r, s, t)
        if way:
            add_way, min_flow = restore_way(way, s, t)
            calc_G_r(G_r, add_way, min_flow)

    return restore_max_flow(G, dic)


def scaling(G, s, t, func=dfs):
    """
    ford fulkerson algorithm: (=improve of run time if the weight large comparing to the len(G.V))
        found max flow in the graph from s to t
          if the weight are irrational numbers the algorithm is not sure the algorithm will converge
          if the weight are rational numbers we can multiple all weight with common factor

    @params:
        @G: graph
        @s: vertex to start
        @t: vertex to end
        @func: function to use for find adding way by default is dfs, can be bfs

    return: dict that contain the flow for every edge

    efficiency: O(log(C_max)*|E|)
    """
    C_max = max(G.E, key=lambda e: e.weight).weight
    theta = 2 ** int(math.log(C_max, 2))
    G_r, dic = init_G_r(G)
    s, t = dic[s], dic[t]
    while theta >= 1:
        while True:
            way = func(G_r, s, t, theta=theta)
            if way:
                add_way, min_flow = restore_way(way, s, t)
                calc_G_r(G_r, add_way, min_flow)
            else:
                theta /= 2
                break

    return restore_max_flow(G, dic)


def init_G_r(G):
    """
    init the Residual graph

    @params:
        @G: graph

    efficiency: O(|E|+|V|)
    """
    G_r = Graph()
    V = [Vertex(name=v.name) for v in G.V]
    dic = {}
    for v1, v2 in zip(G.V, V):
        dic[v1] = v2

    for e in G.E:
        G_r.connect(from_=dic[e.from_], to=dic[e.to], weight=e.weight)

    return G_r, dic


def calc_G_r(G_r, add_way, min_flow):
    """
    update the Residual graph according to the add_way and min flow

    @params:
        @G_r: Residual graph
        @add_way: list of the edges to adding flow
        @min_flow: min weight in add_way (=Bottleneck)

    efficiency: O(|E|+|V|)
    """
    for e in add_way:
        if e.weight - min_flow <= 0:
            G_r.disconnect(e=e)
        else:
            e.weight -= min_flow
        if e.from_ in e.to.Adj:
            e.to.Adj[e.from_].weight += min_flow
        else:
            G_r.connect(from_=e.to, to=e.from_, weight=min_flow)


def restore_way(pi_edges, s, t):
    """
    restore the adding way from the dfs/bfs

    @params:
        @pi_edges: dict of  edges from the bfs/dfs
        @s: vertex to start
        @t: vertex to end

    efficiency: O(|E|) (=worst case)
    """
    add_way, min_flow = [pi_edges[t]], pi_edges[t].weight
    while add_way[0].from_ != s:
        add_way.insert(0, pi_edges[add_way[0].from_])
        min_flow = min(min_flow, add_way[0].weight)

    return add_way, min_flow


def restore_max_flow(G, dic):
    """
    restore the max flow from the Residual graph

    @param:
        @G: Original graph
        @dic: dict of Parallel vertices pairs in G:G_r

    return: dict that contain the flow for every edge

    efficiency: O(|E|)
    """
    flow = {e: None for e in G.E}
    for v in G.V:
        for u in v.Adj:
            if dic[v] in dic[u].Adj:
                flow[v.Adj[u]] = dic[u].Adj[dic[v]].weight
    return flow


def max_pairs(V1, V2, func=dfs):
    """
    ford fulkerson algorithm For maximum pairing:

    :param:
        @G: graph
        @V1,V2: groups of vertex for finding mach
        @func: function to use for find adding way by default is dfs, can be bfs


    :return: list of vertexes pairs for max pairing (=Original vertices)

    efficiency: O(|E|*n) where n is max pairing, in worst case n=|V1|
    """
    G_pairs = Graph()
    dic1, dic2 = {v: Vertex(name=v.name) for v in V1}, {v: Vertex(name=v.name) for v in V2}
    dic, s, t = {**dic1, **dic2}, Vertex(name='s'), Vertex(name='t')
    dic = {dic[k]: k for k in dic}
    for v in dic1:
        G_pairs.connect(from_=s, to=dic1[v], weight=1)
    for v in dic2:
        G_pairs.connect(from_=dic2[v], to=t, weight=1)
    for v in dic1:
        for u in v.Adj:
            if u in dic2:
                G_pairs.connect(from_=dic1[v], to=dic2[u], weight=1)

    max_flow_ = ford_fulkerson(G_pairs, s, t, func=func)
    pairs = []
    for e in max_flow_:
        if max_flow_[e] and e.from_ != s and e.to != t:
            # pairs.append(dic[e.from_].Adj[dic[e.to]])
            pairs.append((dic[e.from_], dic[e.to]))
    return pairs


if __name__ == '__main__':
    V = [Vertex(name=str(i)) for i in range(8)]
    V[0].name, V[1].name, V[2].name, V[3].name, V[4].name, V[5].name, V[6].name, V[7].name = \
        's', '1', '2', '3', '4', '5', '6', 't'
    G = Graph()
    for i in range(8):
        G.V[V[i]] = None
    G.connect(from_=V[0], to=V[1], weight=3)
    G.connect(from_=V[0], to=V[2], weight=3)
    G.connect(from_=V[1], to=V[2], weight=1)
    G.connect(from_=V[1], to=V[3], weight=3)
    G.connect(from_=V[2], to=V[3], weight=1)
    G.connect(from_=V[2], to=V[4], weight=3)
    G.connect(from_=V[3], to=V[4], weight=2)
    G.connect(from_=V[3], to=V[5], weight=3)
    G.connect(from_=V[4], to=V[5], weight=2)
    G.connect(from_=V[4], to=V[6], weight=3)
    G.connect(from_=V[5], to=V[6], weight=3)
    G.connect(from_=V[5], to=V[7], weight=3)
    G.connect(from_=V[6], to=V[7], weight=3)
    print(ford_fulkerson(G, V[0], V[7], func=dfs))
    print(ford_fulkerson(G, V[0], V[7], func=bfs))
    print('--------------------  scalling  ------------------------')
    V = [Vertex(name=str(i)) for i in range(6)]
    V[0].name, V[1].name, V[2].name, V[3].name, V[4].name, V[5].name = 's', '1', '2', '3', '4', 't'
    G = Graph()
    for i in range(6):
        G.V[V[i]] = None
    G.connect(from_=V[0], to=V[1], weight=25)
    G.connect(from_=V[0], to=V[3], weight=20)
    G.connect(from_=V[1], to=V[2], weight=20)
    G.connect(from_=V[2], to=V[3], weight=15)
    G.connect(from_=V[2], to=V[5], weight=16)
    G.connect(from_=V[3], to=V[4], weight=25)
    G.connect(from_=V[4], to=V[5], weight=30)
    print(scaling(G, V[0], V[5], func=dfs))
    print(scaling(G, V[0], V[5], func=bfs))
    print('--------------------  max pairs  ------------------------')
    V1, V2 = [Vertex(name='v' + str(i)) for i in range(5)], [Vertex(name='u' + str(i)) for i in range(5)]
    V1[0].connect(v=V2[0])
    V1[0].connect(v=V2[1])
    V1[1].connect(v=V2[2])
    V1[2].connect(v=V2[2])
    V1[2].connect(v=V2[4])
    V1[3].connect(v=V2[1])
    V1[3].connect(v=V2[3])
    V1[4].connect(v=V2[2])
    V1[4].connect(v=V2[4])
    print(max_pairs(V1, V2))
