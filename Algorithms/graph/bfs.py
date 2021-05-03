from structure.graph_struct import Vertex, Graph, GraphNotAimed
from typing import Tuple


def bfs(G: Graph, s=None) -> Tuple[dict, dict]:
    """
    params:
        v: vertice to start bfs , by default is G.V[0]
    efficiency: O(V+E)
    """
    s = s if s else list(G.V.keys())[0]
    pi, color, layer, Q = {v: None for v in G.V}, {v: 'white' for v in G.V}, {s: 1}, [s if s else G.V[0]]

    while len(Q):
        v = Q.pop(0)
        for u in v.Adj:
            if color[u] == 'white':
                pi[u], color[u], layer[u] = v, 'gray', layer[v] + 1
                Q.append(u)
        color[v] = 'black'

    return pi, layer


def search_circle(G: GraphNotAimed, s=None):
    """
    search circle in not aimed graph

    params:
        v => vertex to start bfs , by default is G.V[0]

    return:
        True/False if there is circle in the graph

    efficiency: O(V+E)
    """
    s = s if s else list(G.V.keys())[0]
    pi, color, layer, Q = {v: None for v in G.V}, {v: 'white' for v in G.V}, {s: 1}, [s]

    while len(Q):
        v = Q.pop(0)
        for u in v.Adj:
            if color[u] == 'white':
                pi[u], color[u], layer[u] = v, 'gray', layer[v] + 1
                Q.append(u)
            elif color[u] == 'gray':
                return True

        color[v] = 'black'

    return False


def two_sided_graph(G: GraphNotAimed):
    """
    return False if the graph is two sides else return the divide into a two-sided graph
        graph is two sides if and only if there is circle in odd length
        and this append only if there is a edge between two vertices in the same layer
    efficiency: O(V+E)
    """
    s = list(G.V.keys())[0]
    pi, color, layer, Q = {v: None for v in G.V}, {v: 'white' for v in G.V}, {s: 1}, [s]

    while len(Q):
        v = Q.pop(0)
        for u in v.Adj:
            if color[u] == 'white':
                pi[u], color[u], layer[u] = v, 'gray', layer[v] + 1
                Q.append(u)
            elif color[u] == 'gray' and layer[v] == layer[u]:
                return False
        color[v] = 'black'

    return layer


if __name__ == '__main__':
    V = [Vertex(name=str(i)) for i in range(5)]
    # r = Edge(from_=V[0], to=V[1], weight=3)
    G = GraphNotAimed(V={v: None for v in V})
    G.connect(from_=V[1], to=V[0], weight=1)
    G.connect(from_=V[2], to=V[1], weight=1)
    G.connect(from_=V[0], to=V[3])
    G.connect(from_=V[2], to=V[0])
    G.connect(from_=V[3], to=V[4])
    G.disconnect(from_=V[2], to=V[0])
    print(G)
    print(bfs(G))
    print(search_circle(G))
    print(two_sided_graph(G))
