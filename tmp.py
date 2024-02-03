# minimum enclosing ball - naive solution using cvxpy
import cvxpy as cp
import numpy as np
from scipy.spatial import ConvexHull, convex_hull_plot_2d

# import pandas as pd
import time
from data import *

# Problem data.
n = 100
d = 2
np.random.seed(1)

def solve(n, d, verbose=False):
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
    constraints = [t >= cp.norm(x - p, 1) for p in hull]
    prob = cp.Problem(objective, constraints)

    t_solve = time.time()
    result = prob.solve()
    t_solve = time.time() - t_solve
    
    constraints = [t >= cp.norm(x - p, 1) for p in A]
    prob = cp.Problem(objective, constraints)
    t_all = time.time()
    result = prob.solve()
    t_all = time.time() - t_all
    
    print(f"n = {n}, d = {d}: hull time = {t_hull:.3f}s, solve time = {t_solve:.3f}s vs. solve time = {t_all:.3f}s")
    if d == 2 and verbose:
        plot_data(A, [np.array(x.value), np.array(x.value)])
    return t_hull + t_solve

# df = pd.DataFrame(['n', 'd', 'time'])

for n in [100, 1000, 10000]:
    for d in [2, 3, 10]:
        ts = solve(n, d, verbose=False)
        # df.append([n, d, time])
