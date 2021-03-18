import copy
from typing import List, Dict, Any


# class Edge:
#
#     def __init__(self, from_, to, weight=1, data=None) -> None:
#         self.from_, self.to, self.weight, self.data = from_, to, weight, data
#
#     def __str__(self):
#         return f'({self.from_.name}->{self.to.name},w={self.weight})'
#
#     def __repr__(self):
#         return f'({self.from_.name}->{self.to.name},w={self.weight})'
#
#
# class Vertex:
#     def __init__(self, name=None, data=None, Adj=None, edges=None):
#         self.Adj: List[Vertex]
#         self.edges: List[Edge]
#         self.name, self.data, self.Adj, self.edges = name, data, Adj, edges
#         self.Adj = Adj if Adj else []
#         self.edges = edges if edges else []
#
#     def connect(self, v=None, weight=1, e=None):
#         if e:
#             if e.to not in self.Adj:
#                 self.Adj.append(e.to)
#                 self.edges.append(e)
#             else:
#                 for e_ in self.edges:
#                     if e_.to == e.v:
#                         e_.weight = weight
#         elif v:
#             if v is self:
#                 return
#             if v not in self.Adj:
#                 self.Adj.append(v)
#                 e = Edge(self, v, weight=weight)
#                 self.edges.append(e)
#             else:
#                 for e_ in self.edges:
#                     if e_.to == v:
#                         e_.weight = weight
#                 return None
#         return e
#
#     def disconnect(self, v):
#         if v in self.Adj:
#             i = self.Adj.index(v)
#             self.Adj.remove(v)
#             del self.edges[i]
#
#     def is_connect(self, v):
#         return v in self.Adj
#
#     def __str__(self) -> str:
#         return f'name: {self.name}, Adj: {self.Adj}, edges: {self.edges}, {self.data}'
#
#     def __repr__(self) -> str:
#         return f'{self.name}'
#
#
# class Graph:
#     E: [Edge]
#     V: [Vertex]
#
#     def __init__(self, E: [Edge] = None, V: [Vertex] = None) -> None:
#         self.E, self.V = E, V
#
#     def connect(self, from_: Vertex = None, to: Vertex = None, weight=1, e: Edge = None):
#         if e:
#             from_, to, weight = e.from_, e.to, e.weight
#         if from_ and to and from_ is not to:
#             e = from_.connect(v=to, weight=weight)
#         else:
#             print('ERROR : must pass @from_,@to OR @e parameters')
#             return
#         from_.connect(v=to, weight=weight)
#         if e and e not in self.E:
#             self.E.append(e)
#         if from_ not in self.V:
#             self.V.append(from_)
#         if to not in self.V:
#             self.V.append(to)
#
#     def disconnect(self, from_: Vertex = None, to: Vertex = None, e: Edge = None):
#         if e:
#             from_, to = e.from_, e.to
#         elif not from_ or not to:
#             print('ERROR : must pass @from_,@to OR @e parameters')
#             return
#         from_.disconnect(v=to)
#         for edg in self.E:
#             if edg.from_ == from_ and edg.to == to:
#                 self.E.remove(edg)
#                 return
#
#     def __str__(self):
#         s = '------------------  graph  -----------------\n'
#         s += 'V=[' + ','.join([v.name for v in self.V]) + ']\n'
#         s += 'E=' + ','.join([str(e) for e in self.E]) + '\n'
#         s += '|V|={} |E|={}'.format(len(self.V), len(self.E)) + '\n'
#         s += '--------------------------------------------'
#         return s
#
#     # return new G' transpose graph
#     def transpose(self):
#         G = Graph()
#         G.V = [Vertex(name=v.name, data=v.data) for v in self.V]
#         G.E = [Edge(G.V[self.V.index(e.to)], G.V[self.V.index(e.from_)], weight=e.weight) for e in self.E]
#         return G
#
#     @property
#     def V(self):
#         return self.__V
#
#     @V.setter
#     def V(self, V):
#         self.__V = V if V is not None else []
#
#     @property
#     def E(self):
#         return self.__E
#
#     @E.setter
#     def E(self, E):
#         self.__E = E if E is not None else []
#         # self.V = []
#         if not E:
#             return
#         for e in E:
#             self.connect(e=e)

class Edge:

    def __init__(self, from_, to, weight=1, data=None) -> None:
        self.from_, self.to, self.weight, self.data = from_, to, weight, data if data else {}

    def __str__(self):
        return f'({self.from_.name}->{self.to.name},w={self.weight})'

    def __repr__(self):
        return f'({self.from_.name}->{self.to.name},w={self.weight})'


class Vertex:
    def __init__(self, name=None, data=None, Adj=None, edges=None):
        self.Adj: Dict[Vertex, Edge]
        self.edges: Dict[Edge, Vertex]
        self.name, self.data, self.Adj, self.edges = name, data, Adj, edges
        self.data = data if data else {}
        self.Adj = Adj if Adj else {}
        self.edges = edges if edges else {}

    def connect(self, v=None, weight=1, e=None):
        if e:
            if e.to not in self.Adj:
                self.Adj[e.to] = e
                self.edges[e] = e.to
            else:
                self.Adj[e.to].weight = weight
                # for e_ in self.edges:
                #     if e_.to == e.v:
                #         e_.weight = weight
        elif v:
            if v is self:
                return
            if v not in self.Adj:
                e = Edge(self, v, weight=weight)
                self.Adj[v] = e
                self.edges[e] = v
            else:
                self.Adj[v].weight = weight
                # for e_ in self.edges:
                #     if e_.to == v:
                #         e_.weight = weight
                return None
        return e

    def disconnect(self, v):
        self.edges.pop(self.Adj.pop(v, None), None)

    def is_connect(self, v):
        return v in self.Adj

    def __str__(self) -> str:
        return f'name: {self.name}, Adj: {self.Adj.keys()}, edges: {self.edges.keys()}, {self.data}'

    def __repr__(self) -> str:
        return f'{self.name}'


class Graph:
    E: Dict[Edge, Any]
    V: Dict[Vertex, Any]

    def __init__(self, V: Dict[Vertex, Any] = None, E: Dict[Edge, Any] = None) -> None:
        self.E, self.V = E, V

    def connect(self, from_: Vertex = None, to: Vertex = None, weight=1, e: Edge = None):
        if e:
            from_, to, weight = e.from_, e.to, e.weight
        if from_ and to and from_ is not to:
            e = from_.connect(v=to, weight=weight)
        else:
            print('ERROR : must pass @from_,@to OR @e parameters')
            return
        from_.connect(v=to, weight=weight)
        if e and e not in self.E:
            self.E[e] = None
        if from_ not in self.V:
            self.V[from_] = None
        if to not in self.V:
            self.V[to] = None

    def disconnect(self, from_: Vertex = None, to: Vertex = None, e: Edge = None):
        if e:
            from_, to = e.from_, e.to
        elif not from_ or not to:
            print('ERROR : must pass @from_,@to OR @e parameters')
            return
        from_.disconnect(v=to)
        for edg in self.E:
            if edg.from_ == from_ and edg.to == to:
                self.E.pop(edg)
                return

    def __str__(self):
        s = '------------------  graph  -----------------\n'
        s += 'V=[' + ','.join([v.name for v in self.V]) + ']\n'
        s += 'E=' + ','.join([str(e) for e in self.E]) + '\n'
        s += '|V|={} |E|={}'.format(len(self.V), len(self.E)) + '\n'
        s += '--------------------------------------------'
        return s

    # return new G' transpose graph
    def transpose(self):
        G = Graph()

        V = {v: Vertex(name=v.name, data=copy.deepcopy(v.data)) for v in self.V.keys()}
        G.V = {v: None for v in V.values()}
        for e in self.E:
            G.connect(from_=V[e.to], to=V[e.from_], weight=e.weight)

        return G

    @property
    def V(self):
        return self.__V

    @V.setter
    def V(self, V):
        self.__V = V if V is not None else {}

    @property
    def E(self):
        return self.__E

    @E.setter
    def E(self, E):
        self.__E = E if E is not None else {}
        # self.V = []
        if not E:
            return
        for e in E:
            self.connect(e=e)


class GraphNotAimed(Graph):

    def __init__(self, E=None, V=None) -> None:
        super().__init__(E=E, V=V)

    def connect(self, from_=None, to=None, weight=1, e=None):
        super().connect(from_=from_, to=to, weight=weight, e=e)
        if e:
            from_, to = e.from_, e.to
        if from_ and to:
            super().connect(from_=to, to=from_, weight=weight)

    def disconnect(self, from_: Vertex = None, to: Vertex = None, e: Edge = None):
        super().disconnect(from_=from_, to=to, e=e)
        if e:
            from_ = e.from_, to = e.to
        super().disconnect(to=from_, from_=to)


if __name__ == '__main__':
    V = [Vertex(name=str(i)) for i in range(10)]
    r = Edge(from_=V[0], to=V[1], weight=3)
    G = Graph(V={v: None for v in V})
    G.connect(from_=V[1], to=V[2], weight=1)
    G.connect(from_=V[2], to=V[1], weight=1)
    G.connect(e=r)
    print(G)
    G.transpose()
    print(G)
    # print(G.transpose())
