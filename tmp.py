# minimum enclosing ball - naive solution using cvxpy
import cvxpy as cp
import numpy as np
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import matplotlib.pyplot as plt

# import pandas as pd
import time
from data import *

# Problem data.
n = 100
d = 2
np.random.seed(1)


def solve_cmp_hull(n, d, verbose=False):
    A = np.random.randn(n, d)
    # A = generate_data(100, 120, 100, (n, d), 0.2)
    t_hull = time.time()
    ch = ConvexHull(A)
    t_hull = time.time() - t_hull
    hull = A[ch.vertices]
    # Construct the problem.
    x = cp.Variable(d)
    t = cp.Variable(1)
    objective = cp.Minimize(t)
    constraints = [t >= cp.norm(x - p, 2) for p in hull]
    prob = cp.Problem(objective, constraints)

    t_solve = time.time()
    result = prob.solve()
    t_solve = time.time() - t_solve

    constraints = [t >= cp.norm(x - p, 2) for p in A]
    prob = cp.Problem(objective, constraints)
    t_all = time.time()
    result = prob.solve()
    t_all = time.time() - t_all

    print(f"n = {n}, d = {d}: hull time = {t_hull:.3f}s, solve time = {t_solve:.3f}s vs. solve time = {t_all:.3f}s")
    if d == 2 and verbose:
        plot_data(A, [np.array(x.value), np.array(x.value)])
    return t_hull + t_solve


def solve(n, d, verbose=False):
    A = np.random.randn(n, d)
    # A = generate_data(100, 120, 100, (n, d), 0.2)
    # 1.
    x = cp.Variable(d)
    t = cp.Variable(1)
    objective = cp.Minimize(t)
    constraints = [cp.SOC(t, x - p) for p in A]
    prob = cp.Problem(objective, constraints)

    t1 = time.time()
    result = prob.solve()
    t1 = time.time() - t1

    return t1

    # 2.
    alpha = cp.Variable(n)
    constraints = [cp.SOC(alpha[i], x - A[i]) for i in range(n)]
    objective = cp.Minimize(cp.norm(alpha, 'inf'))
    prob = cp.Problem(objective, constraints)

    t2 = time.time()
    result = prob.solve()
    t2 = time.time() - t2

    print(f"n = {n}, d = {d}: t1 = {t1:.3f}s, t2 = {t2:.3f}s")
    if d == 2 and verbose:
        plot_data(A, [np.array(x.value), np.array(x.value)])
    return t1, t2


def benchmark(ns=[1000], ds=[2]):
    return np.array([[solve(n, d, verbose=False) for d in ds] for n in ns])
# df = pd.DataFrame(['n', 'd', 'time'])

ds = [1 << i for i in range(12)]
ns = [1 << i for i in range(18)]

ts = [4.996587514877319, 4.62901759147644, 5.1258461475372314, 6.657426118850708, 8.596538066864014, 15.113322973251343, 36.45588445663452, 228.31129813194275, 657.7817392349243, 1521.172934293747, 3664.148594379425, 8594.977717638016]

# ts = benchmark(ns, [8]).reshape(-1)

# ts = [0.00330877304, 0.00296759605, 0.0040910244, 0.0059056282, 0.00996685028, 0.0368821621, 0.0339257717, 0.0660319328, 0.131263256, 0.268102646, 0.551973581, 1.14443636, 2.51777887, 5.79078817, 14.6489606, 43.6893327, 177.118014, 746.847723]
#
# print(ts)
# plt.plot([1 << i for i in range(len(ts))], [np.sqrt(x) for x in ts])
# plt.ylabel("sqrt time (s^1/2)")
# plt.xlabel("n")
# plt.title("cvxpy baseline (d = 8)")
# plt.show()

print(ts)
plt.plot([1 << i for i in range(len(ts))], ts)
plt.ylabel("time (s)")
plt.xlabel("d")
plt.title("cvxpy baseline (n = 1e4)")
plt.show()

