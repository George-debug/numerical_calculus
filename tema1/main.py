
from matplotlib import animation
import matplotlib.pyplot as plt
import time
import random


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
        if (1.1 + u) + u != 1.1 + (u + u):
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


def strassen_mul_power_2(A: list[list], B: list[list]):
    if len(A) == 1:
        return [[A[0][0] * B[0][0]]]

    A11, A12, A21, A22 = split_matrix(A)
    B11, B12, B21, B22 = split_matrix(B)

    P1 = strassen_mul_power_2(add_matrix(A11, A22), add_matrix(B11, B22))
    P2 = strassen_mul_power_2(add_matrix(A21, A22), B11)
    P3 = strassen_mul_power_2(A11, sub_matrix(B12, B22))
    P4 = strassen_mul_power_2(A22, sub_matrix(B21, B11))
    P5 = strassen_mul_power_2(add_matrix(A11, A12), B22)
    P6 = strassen_mul_power_2(sub_matrix(A21, A11), add_matrix(B11, B12))
    P7 = strassen_mul_power_2(sub_matrix(A12, A22), add_matrix(B21, B22))

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

    return C


def next_power_of_2(n: int):
    i = 1
    while i < n:
        i *= 2

    return i


def strassen_mul_square(A: list[list], B: list[list]):
    n = len(A)
    m = next_power_of_2(n)

    A_copy = [[0 for _ in range(m)] for _ in range(m)]
    B_copy = [[0 for _ in range(m)] for _ in range(m)]

    for i in range(n):
        for j in range(n):
            A_copy[i][j] = A[i][j]
            B_copy[i][j] = B[i][j]

    C = strassen_mul_power_2(A_copy, B_copy)

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


def get_test(n):
    A = create_matrix(n)
    B = create_matrix(n)

    start = time.time()
    strassen_mul_square(A, B)
    end = time.time()
    strassen_time = end - start

    start = time.time()
    simple_mul(A, B)
    end = time.time()
    simple_time = end - start

    return strassen_time, simple_time


def create_plot_test():
    global strassen_lines, simple_lines
    strassen_lst = []
    simple_lst = []

    fig, ax = plt.subplots()

    #
    ax.set_xlabel('Matrix size')
    ax.set_ylabel('Time')

    strassen_lines = ax.plot(strassen_lst, label='Strassen', color='red')
    simple_lines = ax.plot(simple_lst, label='Simple', color='blue')

    ax.legend()

    def animate(i):
        global strassen_lines, simple_lines
        n = i + 1
        n *= 1

        strassen_time, simple_time = get_test(n)

        strassen_lst.append((n, strassen_time))
        simple_lst.append((n, simple_time))

        # delete old lines

        for line in strassen_lines:
            line.remove()

        for line in simple_lines:
            line.remove()

        # draw new lines
        strassen_lines = ax.plot(
            *zip(*strassen_lst), label='Strassen', color='red')
        simple_lines = ax.plot(*zip(*simple_lst), label='Simple', color='blue')

    ani = animation.FuncAnimation(fig, animate, interval=1000)

    plt.show()


if __name__ == '__main__':
    # create_plot_test()

    strassen_time, simple_time = get_test(1000)

    print("Strassen time: ", strassen_time)
    print("Simple time: ", simple_time)
