import numpy as np
import time

from quadtree import OneCapacityQuadtree, QuadTree, sqr_distance, distance

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def naive_closest(point, points):
    candidates = points.copy()
    candidates.remove(point)
    return min(candidates, key=lambda p: sqr_distance((p.x, p.y), (point.x, point.y)))


t1 = time.perf_counter()
tree_size = 1024
quad_tree = OneCapacityQuadtree((0, 0), tree_size, tree_size)
nb_points = 1e3
points = [Point(np.random.random()*tree_size, np.random.random()*tree_size) for _ in range(int(nb_points))]
#check that the points are unique
"""duplicate = []
for point in points:
    closest = naive_closest(point, points)
    if distance((point.x, point.y), (closest.x, closest.y)) == 0:
        duplicate.append(point)
for dup in duplicate:
    points.remove(dup)
for point in points:
    closest = naive_closest(point, points)
    assert distance((point.x, point.y), (closest.x, closest.y)) != 0, f"Point {point.x, point.y} is not unique"
nb_points = len(points)"""
t2 = time.perf_counter()

print(f"Time to create {nb_points} points: {round((t2-t1)*1e3, 2)} ms")

t1 = time.perf_counter()
for point in points:
    quad_tree.insert(point)
t2 = time.perf_counter()


print(quad_tree.total_points())

print(f"Time to insert {nb_points} points: {round((t2-t1)*1e3, 2)} ms")

t1 = time.perf_counter()
for point in points:
    closest = quad_tree.closest_naive(point, radius=10)
t2 = time.perf_counter()

print(f"Time to find {nb_points} closest neighbors inside a circle with quadtree: {round((t2-t1)*1e3, 2)} ms")

"""t1 = time.perf_counter()
for point in points:
    closest = naive_closest(point, points)
t2 = time.perf_counter()

print(f"Time to find {nb_points} closest neighbors with naive method: {round((t2-t1)*1e3, 2)} ms")
"""

t1 = time.perf_counter()
for point in points:
    closest = quad_tree.closest_breadth(point)
t2 = time.perf_counter()

print(f"Time to find {nb_points} closest neighbors with breadth quadtree: {round((t2-t1)*1e3, 2)} ms")

t1 = time.perf_counter()
for point in points:
    closest = quad_tree.closest_depth(point)
t2 = time.perf_counter()

print(f"Time to find {nb_points} closest neighbors with ordered depth quadtree: {round((t2-t1)*1e3, 2)} ms")

# now test the correctness of the closest functions
fail1 = 0
fail2 = 0
fail4 = 0

for point in points:
    true_closest = naive_closest(point, points)

    c1 = quad_tree.closest_naive(point, radius=10)
    c2 = quad_tree.closest_breadth(point)
    c4 = quad_tree.closest_depth(point, order_regions=True)

    if c1 is not None:
        assert distance((c1.x, c1.y), (point.x, point.y)) == distance((true_closest.x, true_closest.y), (point.x, point.y)),\
            f"Naive closest: {c1.x, c1.y} != {true_closest.x, true_closest.y}. Distance: {distance((c1.x, c1.y), (point.x, point.y))} != {distance((true_closest.x, true_closest.y), (point.x, point.y))}"
    if distance((c2.x, c2.y), (point.x, point.y)) != distance((true_closest.x, true_closest.y), (point.x, point.y)):
        fail2 += 1
    if distance((c4.x, c4.y), (point.x, point.y)) != distance((true_closest.x, true_closest.y), (point.x, point.y)):
        fail4 += 1

print(f"Naive closest fails: {fail1}")
print(f"Breadth closest fails: {fail2}")
print(f"Depth closest fails: {fail4}")



"""
######
# max capacity = 4 :
######

Time to create 1000.0 points: 0.84 ms
Time to insert 1000.0 points: 5.56 ms

Time to find 1000.0 closest neighbors inside a circle with quadtree: 13.1 ms
Time to find 1000.0 closest neighbors with breadth quadtree: 116.11 ms
Time to find 1000.0 closest neighbors with ordered depth quadtree: 54.12 ms

Naive closest fails: 0
Breadth closest fails: 307
Depth closest fails: 0


######
# max capacity = 1 :
######

Time to create 1000.0 points: 0.85 ms
Time to insert 1000.0 points: 8.47 ms

Time to find 1000.0 closest neighbors inside a circle with quadtree: 14.28 ms
1000
Time to find 1000.0 closest neighbors with breadth quadtree: 137.6 ms
Time to find 1000.0 closest neighbors with ordered depth quadtree: 79.12 ms

Naive closest fails: 0
Breadth closest fails: 0
Depth closest fails: 0


######
# max capacity = true 1 :
######

Time to create 1000.0 points: 0.86 ms
Time to insert 1000.0 points: 8.37 ms

Time to find 1000.0 closest neighbors inside a circle with quadtree: 14.22 ms
Time to find 1000.0 closest neighbors with breadth quadtree: 138.38 ms
Time to find 1000.0 closest neighbors with ordered depth quadtree: 81.17 ms

Naive closest fails: 0
Breadth closest fails: 0
Depth closest fails: 0

"""