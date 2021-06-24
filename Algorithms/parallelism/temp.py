import threading
from threading import Thread, Lock
import numpy as np
import queue


# **********************************  multiply matrix by vector  *************************************
def mult_mat_vec(A, v):
    """
    multiply matrix by vector

    :param A:
    :param v:

    :return: vector of result: u = A @ v

    :complexity: run-time: - T-1 = O(n^2)
                           - T-inf = O(log(n^2))
                 memory: O(n) => vector u
    """
    A, v, u, n = np.array(A), np.array(v), np.zeros((len(v),)), len(v)

    def __mult_mat_vec(i, s, e):
        if s == e:
            u[i] += v[s] * A[i, s]
        if s >= e:
            return

        t1, t2 = Thread(target=__mult_mat_vec, args=(i, s, e // 2)), Thread(target=__mult_mat_vec,
                                                                            args=(i, e // 2 + 1, e))
        t1.start(), t2.start()
        t1.join(), t2.join()

    def _mult_mat_vec(i, j):
        if i == j:
            t = Thread(target=__mult_mat_vec, args=(i, 0, n - 1))
            t.start(), t.join()
        if i >= j:
            return

        t1, t2 = Thread(target=_mult_mat_vec, args=(i, j // 2)), Thread(target=_mult_mat_vec, args=(j // 2 + 1, j))
        t1.start(), t2.start()
        t1.join(), t2.join()

    _mult_mat_vec(0, n - 1)

    return u


# **********************************  min in vector  *************************************
def min_(a, Q: queue.Queue):
    if len(a) == 1:
        return Q.put(a[0])
    elif len(a) == 0:
        return

    t1, t2 = Thread(target=min_, args=(a[:len(a) // 2], Q)), Thread(target=min_, args=(a[len(a) // 2:], Q))
    t1.start(), t2.start()
    t1.join(), t2.join()
    x, y = Q.get(), Q.get()
    return Q.put(min(x, y))

    # ***************************************  p-merge sort  *********************************


def merge_sort(a):
    a = np.array(a)
    __merge_sort(a, 0, a.shape[0] - 1)
    return a


def __merge_sort(a, i, j):
    if i >= j:
        return
    mid = (j - i) // 2
    t1 = Thread(target=__merge_sort, args=(a, i, i + mid))
    t2 = Thread(target=__merge_sort, args=(a, i + mid + 1, j))
    t1.start(), t2.start()
    t1.join(), t2.join()
    merge(a, i, i + mid + 1, j)


def merge(a, i, mid, j):
    b, k = np.empty((j - i + 1,)), 0
    i_, mid_ = i, mid

    while i < mid <= j:
        if a[i] <= a[mid]:
            b[k], i, k = a[i], i + 1, k + 1
        else:
            b[k], mid, k = a[mid], mid + 1, k + 1
    while i < mid_:
        b[k], i, k = a[i], i + 1, k + 1
    while mid <= j:
        b[k], mid, k = a[mid], mid + 1, k + 1

    a[i_:j + 1] = b


# ***************************************  p-sum of vector  *********************************
def sum_(a):
    q = queue.Queue()

    def __sum(i, j):
        if i == j:
            return q.put(a[i])
        if i >= j:
            return

        # mid = (i+j) // 2
        # print(i,i+mid)
        # print(mid+1,j)
        # t1 = Thread(target=__sum, args=(i, i + mid))
        # t2 = Thread(target=__sum, args=(mid+1, j))
        # t1.start(),t2.start()
        # t1.join(),t2.join()
        # x,y=q.get(),q.get()
        # q.put(x+y)
    # __sum(0,len(a)-1)
    return q.get()


# ***************************************  main  *********************************
if __name__ == '__main__':
    """
    x = 0
lock = Lock()


def f():
    lock.acquire()
    global x
    x += 5
    lock.release()
    # while 1:
    #     r = 0

    """
    print('-----------------  p-merge sort  -----------------------')
    a = [2, 3, 1, 0, -9]
    a_ = merge_sort(a)
    print('--------------  p-min  -----------------------')
    print(a_)
    Q = queue.Queue()
    min_(np.array([-9, -90, 8, 7, 9]), Q)
    print(Q.get())
    print('-----------------  multiply matrix by vector  -----------------------')
    A, v = np.array([[1, 2, 4], [2, 3, 4], [5, 6, 7]]), np.array([1, 2, 3])
    print(A @ v)
    print(mult_mat_vec(A, v))
    print('-----------------  sum of vector  -----------------------')
    # print(sum_(np.array([1, 2, 3, 4, 5, 6])))
