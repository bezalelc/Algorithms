from Algorithms.structure.graph_struct import Graph, Vertex
from typing import Union, Tuple


def dfs(G: Graph, s=None) -> Tuple[dict, dict, dict]:
    """
    dfs algorithm

    @param:
        @G: graph
        @s: vertex to start from , by default is G.V[0]

    return:
        return (pi, d, f) => (dict of pi, dict of enter time, dict of exit time)

    efficiency: O(V+E)
    """
    time = 0
    pi, color, d, f = {v: None for v in G.V}, {v: 'white' for v in G.V}, {}, {}

    def dfs_visit(v: Vertex):
        nonlocal time
        color[v], time, d[v] = 'gray', time + 1, time + 1
        for u in v.Adj:
            if color[u] == 'white':
                pi[u] = v
                dfs_visit(u)
        color[v] = 'black'
        time, f[v] = time + 1, time + 1

    if s:
        dfs_visit(s)
    for v in G.V:
        if color[v] == 'white':
            dfs_visit(v)
    return pi, d, f


def search_circle(G: Graph) -> Union[bool, dict]:
    """
    search circle with dfs

    @param:
        @G: graph

    return:
        return: dict of exit time if circle not found or False if circle found

    efficiency: O(V+E)
    """

    time, circle = 0, False
    color, f = {v: 'white' for v in G.V}, {}

    def dfs_visit(v: Vertex):
        nonlocal time, circle
        color[v], time = 'gray', time + 1
        for u in v.Adj:
            if color[u] == 'white':
                dfs_visit(u)
            elif color == 'gray':
                circle = False
                return

        color[v] = 'black'

        time, f[v] = time + 1, time + 1

    for v in G.V:
        if color[v] == 'white':
            if not circle:
                dfs_visit(v)
            else:
                return False
    return f


def forest(G, v=None):
    """
    Classification of the edges in G:
        using dfs

    param:
        @G: graph
        @v: vertex to start dfs , by default is G.V[0]

    return:
        return_edge, cross_edge, forest_edge, forward_edge, dict of pi, dict of enter time, dict of exit time

    efficiency: O(V+E)
    """
    v = v if v else V[0]
    time = 0
    pi, color, d, f = {v: None for v in G.V}, {v: 'white' for v in G.V}, {}, {}
    return_edge, cross_edge, forest_edge, forward_edge = [], [], [], []

    def dfs_visit(v: Vertex):
        nonlocal time
        color[v], time, d[v] = 'gray', time + 1, time + 1
        for e in v.edges:
            if color[e.to] == 'white':
                forest_edge.append(e)
                # pi[e.to] = v
                dfs_visit(e.to)
            elif color[e.to] == 'gray':
                return_edge.append(e)
            elif color[e.to] == 'black':
                if d[e.to] > d[v]:
                    forest_edge.append(e)
                elif d[e.to] < d[v]:
                    cross_edge.append(e)
        color[v] = 'black'
        time, f[v] = time + 1, time + 1

    dfs_visit(v)
    for v in G.V:
        if color[v] == 'white':
            dfs_visit(v)
    return return_edge, cross_edge, forest_edge, forward_edge, pi, d, f


def topology(G: Graph) -> Union[list, None]:
    """
    efficiency: O(V+E)
    """
    f = search_circle(G)
    if not f:
        print('there is a circle in the graph')
        return
    topology_order = list(f.items())
    topology_order.sort(key=lambda item: item[1], reverse=True)
    return [item[0] for item in topology_order]


def sccg(G: Graph):
    """
    strongly connected component graph
    efficiency: O(V+E)
    """
    _, _, f = dfs(G)
    f = [(item[0], item[1]) for item in f.items()]
    f.sort(key=lambda item: item[1], reverse=True)
    for v, i in zip(f, range(len(G.V))):
        v[0].data['i'] = i
    G_t = G.transpose()

    def dfs_visit(v: Vertex):
        color[v] = 'gray'
        for u in v.Adj:
            if color[u] == 'white':
                tie_well[len(tie_well) - 1].append(f[u.data['i']][0])
                dfs_visit(u)

    color, V, tie_well = {v: 'white' for v in G_t.V}, sorted(G_t.V.keys(), key=lambda v: v.data['i']), []
    for v in V:
        if color[v] == 'white':
            tie_well.append([f[v.data['i']][0]])
            dfs_visit(v)

    return tie_well


def roots(G):
    _, _, f = dfs(G)
    s = max(f.items(), key=lambda item: item[1])[0]
    pi, _, _ = dfs(G, s=s)
    for v, p in pi.items():
        if not p and v is not s:
            return False

    G_scc = tie_well_graph(G)
    return topology(G_scc)[0]


def leaves(G):
    return roots(G.transpose())


def tie_well_graph(G):
    """
    efficiency: O(V+E)
    """
    tie_well = sccg(G)
    G_scc = Graph()

    for v_group, i in zip(tie_well, range(len(tie_well))):
        v = Vertex(name=str(v_group), data={'group': v_group})
        for v1 in v_group:
            v1.data['group'] = v
    for v in G.V:
        for u in v.Adj:
            if v.data['group'] != u.data['group']:
                G_scc.connect(from_=v.data['group'], to=u.data['group'], weight=v.Adj[u].weight)

    return G_scc


if __name__ == '__main__':
    V = [Vertex(name=str(i)) for i in range(5)]
    # r = Edge(from_=V[0], to=V[1], weight=3)
    G = Graph(V={v: None for v in V})
    G.connect(from_=V[1], to=V[2], weight=1)
    G.connect(from_=V[2], to=V[1], weight=1)
    # G.connect(e=r)
    G.connect(from_=V[3], to=V[2])
    G.connect(from_=V[4], to=V[0])
    G.connect(from_=V[3], to=V[0])
    G.connect(from_=V[1], to=V[4])
    dfs(G, list(G.V.keys())[0])
    search_circle(G)
    forest(G, list(G.V.keys())[0])
    # print(G.V.keys()[0])
    topology(G)
    print(topology(G))
    print(sccg(G))
    print(leaves(G))
    print(tie_well_graph(G))
