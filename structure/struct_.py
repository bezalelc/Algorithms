class Node:
    def __init__(self, name=None, data=None, prev=None, next_=None):
        self.name = name, self.data, self.prev, self.next_ = name, data, prev, next_

    def __str__(self) -> str:
        return f'data: {self.data}, prev: {None if self.prev is None else self.prev.data}, ' \
               f'next: {None if self.next_ is None else self.next_.data} '


if __name__ == '__main__':
    pass
