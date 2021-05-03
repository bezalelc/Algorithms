class ListNode:
    def __init__(self, name=None, data=None, prev=None, next_=None):
        self.name = name, self.data, self.prev, self.next_ = name, data, prev, next_

    def __str__(self) -> str:
        return f'name: {self.name}, prev: {None if self.prev is None else self.prev.data}, next: {None if self.next_ is None else self.next_.data}, data: {self.data}'

    def __repr__(self):
        return f'{self.data}'


# class LinkedList:
#     """
#        linked list: A One-way linked list
#
#        override methods:
#            __str__(): => print as list format
#            __repr__(): => print details of the nodes
#            __iadd__(): => add ListNode/data to the list
#            __add__(list_other): => add list other to the end of the self list
#            __iter__() , __next__(): useful for loops , iter from head to the last
#
#        @staticmethod:
#            arr_to_list(array): => convert array to linked list in the order of the array
#
#        """
#
#     def __init__(self, head=None, name=None, other=None):
#         if other:  # copy constructor
#             pass
#         else:
#             self.head, self.name, self.n = head, name, 0
#
#     @property
#     def head(self):
#         return self.__head
#
#     @head.setter
#     def head(self, head):
#         self.__head = head
#
#     def __str__(self):
#         s = 'head'
#         for node in self:
#             s += '->' + repr(node)
#         s += '->last'
#         return s
#
#     def __repr__(self) -> str:
#         p = self.head
#         s = "--------------------------  list details  ----------------\n"
#         for node in self:
#             s += node.__str__() + "\n"
#         s += "---------------------------  n = {}  -----------------\n".format(self.n)
#         return s
#
#     def __add__(self, list_other):
#         self.last.next_ = list_other.head
#         self.last = list_other.last
#         return self
#
#     def __iadd__(self, data, add_to_end=False):
#         data = data if isinstance(data, ListNode) else ListNode(data=data)
#         self.n += 1
#         if self.head is None:
#             self.head = self.last = data
#         elif add_to_end:
#             data.next_ = self.head
#             self.head.prev = data
#             self.head = data
#         else:
#             data.prev = self.last
#             self.last.next_ = data
#             self.last = data
#         return self
#
#     def __next__(self):
#         if self.itr:
#             p = self.itr
#             self.itr = self.itr.next_
#             return p
#         else:
#             raise StopIteration
#
#     def __iter__(self):
#         self.itr = self.head
#         return self
#
#     @staticmethod
#     def arr_to_list(array):
#         if len(array) == 0:
#             return
#         l = TwoWayList()
#         for data in array:
#             l += data
#         return l


class TwoWayList:
    """
    linked list: A two-way linked list

    override methods:
        __str__(): => print as list format
        __repr__(): => print details of the nodes
        __iadd__(): => add ListNode/data to the list
        __add__(list_other): => add list other to the end of the self list
        __iter__() , __next__(): useful for loops , iter from head to the last

    @staticmethod:
        arr_to_list(array): => convert array to linked list in the order of the array

    """

    def __init__(self, head=None, last=None, name=None, other=None):
        # super().__init__(head, name, other)
        # self.last = last
        if other:  # copy constructor
            pass
        else:
            self.head, self.last, self.name, self.n = head, last, name, 0

    # @property
    # def head(self):
    #     return self.__head
    #
    # @head.setter
    # def head(self, head):
    #     self.__head = self.last = head if (not head or isinstance(head, ListNode)) else ListNode(data=head)

    # = self.__head
    # self.__set_head_last(head=self.__head)

    # @property
    # def last(self):
    #     return self.__last
    #
    # @last.setter
    # def last(self, last):
    #     self.__last = last if not last or isinstance(last, ListNode) else ListNode(data=last)
    #     # self.__set_head_last(last=self.__last)

    # def __set_head_last(self, head=None, last=None):
    #     if head:
    #         self.last = head
    #     if last:
    #         self.head = last

    def __str__(self):
        s = 'head'
        for node in self:
            s += '->' + repr(node.data)
        s += '->last'
        return s

    def __repr__(self) -> str:
        p = self.head
        s = "--------------------------  list details  ----------------\n"
        for node in self:
            s += node.__str__() + "\n"
        s += "---------------------------  n = {}  -----------------\n".format(self.n)
        return s

    def __add__(self, list_other):
        self.last.next_ = list_other.head
        self.last = list_other.last
        return self

    def __iadd__(self, data, add_to_end=False):
        data = data if isinstance(data, ListNode) else ListNode(data=data)
        self.n += 1
        if self.head is None:
            self.head = self.last = data
        elif add_to_end:
            data.next_ = self.head
            self.head.prev = data
            self.head = data
        else:
            data.prev = self.last
            self.last.next_ = data
            self.last = data
        return self

    def __next__(self):
        if self.itr:
            p = self.itr
            self.itr = self.itr.next_
            return p
        else:
            raise StopIteration

    def __iter__(self):
        self.itr = self.head
        return self

    @staticmethod
    def arr_to_list(array):
        if len(array) == 0:
            return
        l = TwoWayList()
        for data in array:
            l += data
        return l


if __name__ == '__main__':
    import copy

    a = [1, 3, 4]
    # # l = List(7)
    lis = TwoWayList.arr_to_list([3, 4, 7, 8, 9])
    lis1 = TwoWayList.arr_to_list([3, 4, 7, 8, 9])
    lis2 = lis + lis1
    print(lis2)
    # n1 = ListNode(data=8)
    # for node in lis:
    #     print(node)
    # lis.head = n1
    # print(lis)
    #
    # lis1 = copy.deepcopy(lis)
    # print(lis1)

    # xs = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    # zs = copy.deepcopy(xs)
    # print(xs)
    # zs[1] = 1
    # print('after:\n')
    # print(xs)
    # print(zs)
