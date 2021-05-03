from structure.graph_struct import GraphNotAimed, Vertex
from structure.list_struct import TwoWayList
from structure.heap import BinaryHeap


def prim(G, restore=False):
    """
    prim algorithm:
        if |E| * 0.75  < |V|^2 : the struct will be min heap
        else:                    the struct will be array

    @params:
        @G: graph
        @restore: if restore the solution

    return:
        return pi dict
        if restore=True the mst graph (=Minimal spreadsheet graph) will be return also

    efficiency:
        if |E| * 0.75  < |V|^2: => O(Vlog V)
        if |E| * 0.75  > |V|^2: => O(E) = O(|V|^2)
    """
    struct = 'heap' if len(G.E) < (len(G.V) ** 2) * 0.75 else 'array'
    pi = {v: None for v in G.V}
    for v in G.V:
        v.data = {'key': float('inf'), 'in heap': True}
    list(G.V.keys())[0].data['key'] = 0
    key = lambda v: v.data['key']
    Q = BinaryHeapPrim(arr=G.V, compare=min, key=key) if struct == 'heap' else [v for v in G.V]

    while struct == 'heap' and Q.arr or struct == 'array' and Q:
        v, v.data['in heap'] = Q.extract() if struct == 'heap' else Q.pop(Q.index(min(Q, key=key))), False
        for e in v.edges:
            if e.to.data['in heap'] and e.weight < e.to.data['key']:
                pi[e.to], e.to.data['key'] = v, e.weight
                if struct == 'heap':
                    Q.add(Q.pop(e.to.data['i']))

    if restore:
        V = {v: Vertex(name=v.name, data=v.data) for v in G.V}
        G_mst = GraphNotAimed()
        for v, u in pi.items():
            if u:
                G_mst.connect(from_=V[v], to=V[u], weight=v.data['key'])
        return pi, G_mst

    return pi


def kruskal(G, restore=False):
    """
       kruskal algorithm:

       @params:
           @G: graph
           @restore: if restore the solution

       return:
           return pi dictionary
           if restore=True the mst graph (=Minimal spreadsheet graph) will be return also

       efficiency:
           the sort of the edges: O(|E|log(|E|)) , the loop: O(|V|log(|V|) => total: O(|E|log(|E|)
           if the edges already sorted kruskal is fasted than prim
       """

    def union(e):
        v, u = set_id[e.from_], set_id[e.to]
        v, u = (v, u) if sets[v].n > sets[u].n else (u, v)
        for node in sets[u]:
            set_id[node.data] = v
        sets[v] = sets[v] + sets[u]
        sets.pop(u)

    E_sor, sets, set_id = sorted(G.E, key=lambda e: e.weight), {v: TwoWayList() for v in G.V}, {v: v for v in G.V}
    for v, lis in sets.items():
        lis += v

    A = []
    for e in E_sor:
        if set_id[e.from_] is not set_id[e.to]:
            A.append(e)
            union(e)

    if restore:
        V = {v: Vertex(name=v.name, data=v.data) for v in G.V}
        G_mst = GraphNotAimed()
        for e in A:
            G_mst.connect(from_=V[e.from_], to=V[e.to], weight=e.weight)
        return A, G_mst

    return A


class BinaryHeapPrim(BinaryHeap):
    """
    this class is for prim algorithm only , the vertex in this class contain 'i' => index of the object
        in the heap so we can add/pop element in O(1) if we have the element itself (=instead of O(n))
        the 'i' stored in dict in Vertex.data['i']
    """

    def __init__(self, arr: [Vertex] = None, compare=max, key=None) -> None:
        if arr:
            for v, i in zip(arr, range(len(arr))):
                if v.data:
                    v.data['i'] = i
                else:
                    v.data = {'i': i}
        super().__init__(arr, compare, key)

    # efficiency: O(log n)
    def heapify_up(self, i):
        while i > 0:
            i_max = self.arr.index(self.compare(self.arr[i], self.arr[self.father(i)], key=self.key))
            if i == i_max:
                # added for prim ----------------------------------
                temp = self.arr[i].data['i']
                self.arr[i].data['i'] = self.arr[self.father(i)].data['i']
                self.arr[self.father(i)].data['i'] = temp
                # added for prim ----------------------------------
                self.arr[i], self.arr[self.father(i)], i = self.arr[self.father(i)], self.arr[i], self.father(i)
            else:
                return

    # efficiency: O(log n)
    def heapify_down(self, i):
        while i < len(self.arr) // 2:
            r, l = self.left(i), self.right(i)  # indexes of right and left suns
            i_max = self.arr.index(self.compare([self.arr[x] for x in [i, r, l] if x is not False], key=self.key))
            if i_max != i:
                # added for prim ----------------------------------
                temp = self.arr[i].data['i']
                self.arr[i].data['i'] = self.arr[i_max].data['i']
                self.arr[i_max].data['i'] = temp
                # added for prim ----------------------------------
                self.arr[i], self.arr[i_max], i = self.arr[i_max], self.arr[i], i_max
            else:
                return

    # efficiency: O(log n)
    def pop(self, i):
        self.arr[i], self.arr[len(self.arr) - 1] = self.arr[len(self.arr) - 1], self.arr[i]
        # added for prim ----------------------------------
        temp = self.arr[i].data['i']
        self.arr[i].data['i'] = self.arr[len(self.arr) - 1].data['i']
        self.arr[len(self.arr) - 1].data['i'] = temp
        # added for prim ----------------------------------
        data = self.arr.pop()
        if i < len(self.arr):
            self.heapify_down(i)
        return data

    # efficiency: O(log n)
    def extract(self):
        return self.pop(0)

    # efficiency: O(log n)
    def add(self, data):
        self.arr.append(data)
        # added for prim ----------------------------------
        self.arr[len(self.arr) - 1].data['i'] = len(self.arr) - 1
        # added for prim ----------------------------------
        self.heapify_up(len(self.arr) - 1)


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
    # print(G)
    print(prim(G, restore=True)[1])
    print(kruskal(G, restore=True))
    t = kruskal(G)

