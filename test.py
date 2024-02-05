from quadtree import QuadTree

import numpy as np
import time

npoints = 1e3

tree = QuadTree(np.array([0, 0]), 1, 1, capacity=1)

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.vect = np.array([x, y])

points = [Point(np.random.random(), np.random.random()) for _ in range(int(npoints))]

t1 = time.perf_counter()
for point in points:
    tree.insert(point)
t2 = time.perf_counter()
print(f"Time to insert {npoints} points: {round((t2-t1)*1e3, 2)} ms")

t1 = time.perf_counter()
for point in points:
    closest = tree.closest_naive(point, radius=0.1)
t2 = time.perf_counter()
print(f"Time to find {npoints} closest neighbors inside a circle with quadtree: {round((t2-t1)*1e3, 2)} ms")

t1 = time.perf_counter()
for point in points:
    closest = tree.closest_depth(point)
t2 = time.perf_counter()
print(f"Time to find {npoints} closest neighbors with quadtree: {round((t2-t1)*1e3, 2)} ms")

t1 = time.perf_counter()
for point in points:
    closest = min(points, key=lambda p: np.linalg.norm(p.vect - point.vect))
t2 = time.perf_counter()
print(f"Time to find {npoints} closest neighbors without quadtree: {round((t2-t1)*1e3, 2)} ms")