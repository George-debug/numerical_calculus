
from threading import Thread
from matplotlib import animation
import matplotlib.pyplot as plt
import time
import random

import numpy as np


def get_precition():
    i = 1
    while True:
        u = 10**(-i)
        if 1 + u == 1:
            return i

        i += 1


def verify_add_asociativity():
    i = 1
    while True:
        u = 10**(-i)
        if (1 + u) + u != 1 + (u + u):
            return (1, u, u)

        i += 1


def verify_mul_asociativity():
    i = 1
    while True:
        u = 10**(-i)
        if (1.1 * u) * u != 1.1 * (u * u):
            return (1.1, u, u)

        i += 1


def print_matrix(A: list[list]):
    for i in range(len(A)):
        for j in range(len(A[i])):
            print(A[i][j], end=' ')
        print()

    print()


def split_matrix(A: list[list]):
    n = len(A)
    half_n = n // 2

    A11 = [[A[i][j] for j in range(half_n)] for i in range(half_n)]
    A12 = [[A[i][j] for j in range(half_n, n)] for i in range(half_n)]
    A21 = [[A[i][j] for j in range(half_n)] for i in range(half_n, n)]
    A22 = [[A[i][j] for j in range(half_n, n)] for i in range(half_n, n)]

    return A11, A12, A21, A22


def add_matrix(A: list[list], B: list[list]):
    return [[A[i][j] + B[i][j] for j in range(len(A))] for i in range(len(A))]


def sub_matrix(A: list[list], B: list[list]):
    return [[A[i][j] - B[i][j] for j in range(len(A))] for i in range(len(A))]


def strassen_mul_power_2(A: list[list], B: list[list], n_min: int):

    if len(A) <= n_min:
        return simple_mul(A, B)

    # print("start", len(A))

    A11, A12, A21, A22 = split_matrix(A)
    B11, B12, B21, B22 = split_matrix(B)

    P1 = strassen_mul_power_2(add_matrix(
        A11, A22), add_matrix(B11, B22), n_min)
    P2 = strassen_mul_power_2(add_matrix(A21, A22), B11, n_min)
    P3 = strassen_mul_power_2(A11, sub_matrix(B12, B22), n_min)
    P4 = strassen_mul_power_2(A22, sub_matrix(B21, B11), n_min)
    P5 = strassen_mul_power_2(add_matrix(A11, A12), B22, n_min)
    P6 = strassen_mul_power_2(sub_matrix(
        A21, A11), add_matrix(B11, B12), n_min)
    P7 = strassen_mul_power_2(sub_matrix(
        A12, A22), add_matrix(B21, B22), n_min)

    C11 = add_matrix(sub_matrix(add_matrix(P1, P4), P5), P7)
    C12 = add_matrix(P3, P5)
    C21 = add_matrix(P2, P4)
    C22 = add_matrix(sub_matrix(add_matrix(P1, P3), P2), P6)

    C = [[0 for _ in range(len(A))] for _ in range(len(A))]
    for i in range(len(C) // 2):
        for j in range(len(C) // 2):
            C[i][j] = C11[i][j]
            C[i][j + len(C) // 2] = C12[i][j]
            C[i + len(C) // 2][j] = C21[i][j]
            C[i + len(C) // 2][j + len(C) // 2] = C22[i][j]

    # print("end", len(A))

    return C


def next_power_of_2(n: int):
    i = 1
    while i < n:
        i *= 2

    return i


def strassen_mul_square(A: list[list], B: list[list], n_min=4):
    n = len(A)
    m = next_power_of_2(n)

    A_copy = [[0 for _ in range(m)] for _ in range(m)]
    B_copy = [[0 for _ in range(m)] for _ in range(m)]

    for i in range(n):
        for j in range(n):
            A_copy[i][j] = A[i][j]
            B_copy[i][j] = B[i][j]

    C = strassen_mul_power_2(A_copy, B_copy, n_min=n_min)

    # Remove padding
    for i in range(n):
        C[i] = C[i][:n]

    return [[C[i][j] for j in range(n)] for i in range(n)]


my_eps = 10**(-10)


def equal_matrix(A: list[list], B: list[list]):
    for i in range(len(A)):
        for j in range(len(A[i])):
            if abs(A[i][j] - B[i][j]) > my_eps:
                return False

    return True


matrix_range = (1, 3)


def simple_mul(A: list[list], B: list[list]):
    n = len(A)
    return [[sum([A[i][k] * B[k][j] for k in range(n)])
             for j in range(n)] for i in range(n)]


def create_matrix(n: int):
    return [[random.uniform(*matrix_range) for _ in range(n)]
            for _ in range(n)]


strassen_lines = []
simple_lines = []
np_lines = []


def get_test(n, n_min=1):
    A = create_matrix(n)
    B = create_matrix(n)

    start = time.time()
    strassen_mul_square(A, B, n_min=n_min)
    end = time.time()
    strassen_time = end - start

    start = time.time()
    simple_mul(A, B)
    end = time.time()
    simple_time = end - start

    start = time.time()
    np.matmul(A, B)
    end = time.time()
    np_time = end - start

    return strassen_time, simple_time, np_time


class PlotAnimateWrapper:
    def __init__(self, animate_func, max_counter):
        self.animate_func = animate_func
        self.runningThread = None
        self.counter = 0
        self.max_counter = max_counter

    def __call__(self, *args, **kwargs):
        if self.counter >= self.max_counter:
            return None
        if self.runningThread is None or not self.runningThread.is_alive():
            if len(args) > 0:
                args = list(args)
                args[0] = self.counter
                args = tuple(args)
                self.counter += 1
            self.runningThread = Thread(
                target=self.animate_func, args=args, kwargs=kwargs)
            self.runningThread.start()
            return self.runningThread
        return self.runningThread


def create_plot_test(max_i=10, n_min=8):
    global strassen_lines, simple_lines
    strassen_lst = []
    simple_lst = []
    np_lst = []

    fig, ax = plt.subplots()

    #
    ax.set_xlabel('Matrix size')
    ax.set_ylabel('Time')

    strassen_lines = ax.plot(strassen_lst, label='Strassen', color='red')
    simple_lines = ax.plot(simple_lst, label='Simple', color='blue')
    np_lines = ax.plot(np_lst, label='Numpy', color='green')

    ax.legend()

    def animate(i):
        global strassen_lines, simple_lines, np_lines
        n = i + 1
        # n *= 10
        # n = 2**(i//2) if i % 2 == 0 else 2**(i//2) + 1
        # n = 2**i

        strassen_time, simple_time, np_time = get_test(n, n_min=n_min)

        strassen_lst.append((n, strassen_time))
        simple_lst.append((n, simple_time))
        np_lst.append((n, np_time))

        # delete old lines

        for line in strassen_lines:
            line.remove()

        for line in simple_lines:
            line.remove()

        for line in np_lines:
            line.remove()

        # draw new lines
        strassen_lines = ax.plot(
            *zip(*strassen_lst), label='Strassen', color='red')
        simple_lines = ax.plot(*zip(*simple_lst), label='Simple', color='blue')
        np_lines = ax.plot(*zip(*np_lst), label='Numpy', color='green')

    animateWrapper = PlotAnimateWrapper(animate, max_i)

    ani = animation.FuncAnimation(fig, animateWrapper, interval=1000)

    plt.show()


if __name__ == '__main__':
    create_plot_test(max_i=100, n_min=8)

    x, y, z = verify_add_asociativity()
    sum1 = (x + y) + z
    sum2 = x + (y + z)
    print(f"( {x} + {y} ) + {z} != {x} + ( {y} + {z} ) ")
    print(sum1, "!= ", sum2)

    x, y, z = verify_mul_asociativity()
    prod1 = (x * y) * z
    prod2 = x * (y * z)
    print(f"( {x} * {y} ) * {z} != {x} * ( {y} * {z} ) ")
    print(prod1, "!= ", prod2)

    # strassen_time, simple_time = get_test(1000, n_min=4)

    # print("Strassen time: ", strassen_time)
    # print("Simple time: ", simple_time)
