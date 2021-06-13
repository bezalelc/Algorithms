from typing import Union, Tuple
from Algorithms.structure.graph_struct import Vertex, Graph, Edge, GraphNotAimed
import dfs, mst


def f(G: Graph, s: Vertex):
    _, G_mst = mst.prim(G, True)
    # print(G_mst)
    # print(dfs.dfs(G, s))
    # print(dfs.dfs(G_mst, s))


def wvc_1(G: Graph):
    min_wvc = float('inf')



def wvc_2():
    pass


def wvc_3():
    pass


def f2():
    pass


if __name__ == '__main__':
    V = [Vertex(name=str(i)) for i in range(6)]
    V[0].name, V[1].name, V[2].name, V[3].name, V[4].name, V[5].name = 'a', 'b', 'c', 'd', 'e', 'f'
    # r = Edge(from_=V[0], to=V[1], weight=3)
    G = GraphNotAimed(V={v: None for v in V})
    G.connect(from_=V[0], to=V[1], weight=4)
    G.connect(from_=V[0], to=V[4], weight=3)
    G.connect(from_=V[0], to=V[3], weight=2)
    G.connect(from_=V[1], to=V[2], weight=5)
    G.connect(from_=V[1], to=V[4], weight=4)
    G.connect(from_=V[1], to=V[5], weight=6)
    G.connect(from_=V[2], to=V[5], weight=1)
    G.connect(from_=V[3], to=V[4], weight=6)
    G.connect(from_=V[4], to=V[5], weight=8)
    # ---------------------
    G.connect(from_=V[0], to=V[2], weight=8)
    G.connect(from_=V[0], to=V[5], weight=8)
    G.connect(from_=V[1], to=V[0], weight=8)
    G.connect(from_=V[1], to=V[3], weight=8)
    G.connect(from_=V[2], to=V[3], weight=8)
    G.connect(from_=V[2], to=V[4], weight=8)
    # ---------------------
    f(G, V[0])
    print('---------------------  wvc  ------------------------')
    G = GraphNotAimed()
    V = [Vertex(name=str(i)) for i in range(4)]
    # G.connect(V[0],V[1],weight=)
    print('---------------------  f2  ------------------------')
