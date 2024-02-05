"""
Implementation of a quadtree data structure.
A quadtree is a spatial tree with the following properties:
    - Each node represents a rectangular region of space.
    - Each node has a maximum capacity of points.
    - If a node exceeds its capacity, it splits into four subnodes.

This allows for algorithms with extremely good complexity to work with space related problems, like finding
    points in a given area or finding the closest point to a given point by not even considering any point that
    has no chance of being interesting.

"""
import numpy as np

QUADTREE_MAX_CAPACITY = 1

def sqr_distance(p1, p2):
    """
    Return the distance between two points without taking the square root.
    """
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

def dist_to_rect_squared(rect, point):
    clamped_x = min(max(point[0], rect.top_left[0]), rect.top_left[0] + rect.width)
    clamped_y = min(max(point[1], rect.top_left[1]), rect.top_left[1] + rect.height)
    clamped = (clamped_x, clamped_y)
    return sqr_distance(point, clamped)

class QuadTree:
    def __init__(self, top_left, height, width, capacity=QUADTREE_MAX_CAPACITY):
        """
        Objects stored in a quadtree must have a vect attribute that is a tuple of two floats.
        """
        self.top_left = top_left
        self.height = height
        self.width = width

        # If false, this node is a leaf. Otherwise, it has four children.
        self.divided = False

        self.capacity = capacity
        self.points = []
        self.representative = None
        self.empty = True

        self.NE = None
        self.NW = None
        self.SE = None
        self.SW = None

    def depth(self):
        """
        Return the depth of this quadtree.
        """
        if not self.divided:
            return 1
        else:
            return 1 + max(self.NE.depth(), self.NW.depth(), self.SE.depth(), self.SW.depth())
    
    def total_points(self):
        """
        Return the total number of points in this quadtree.
        """
        if not self.divided:
            return len(self.points)
        else:
            return (self.NE.total_points() + self.NW.total_points() +
                    self.SE.total_points() + self.SW.total_points())
        
    def total_leafs(self):
        """
        Return the total number of leafs in this quadtree.
        """
        if not self.divided:
            return 1
        else:
            return (self.NE.total_leafs() + self.NW.total_leafs() +
                    self.SE.total_leafs() + self.SW.total_leafs())
    
    def contains(self, point):
        """
        Return true if this quadtree contains the given point.
        """
        x, y = self.top_left
        px, py = point.vect
        
        return (px >= x and px < x + self.width and
                py >= y and py < y + self.height)

    def subdivide(self):
        """
        Subdivide this quadtree into four subnodes.
        """
        x, y = self.top_left
        h = self.height / 2
        w = self.width / 2

        self.NE = QuadTree((x + w, y), h, w, self.capacity)
        self.NW = QuadTree((x, y), h, w, self.capacity)
        self.SE = QuadTree((x + w, y + h), h, w, self.capacity)
        self.SW = QuadTree((x, y + h), h, w, self.capacity)

        for point in self.points:
            self.NE.insert(point)
            self.NW.insert(point)
            self.SE.insert(point)
            self.SW.insert(point)

        self.divided = True
    
    def merge(self):
        """
        Try to merge the children of this quadtree.
        """
        if not self.divided:
            return
        #counting all points every time is very costly, so we count only when all four children are leaves
        if self.NE.divided or self.NW.divided or self.SE.divided or self.SW.divided:
            return
        if self.total_points() > self.capacity:
            return
        
        # theoretically, there is no need to recursively merge children as they are merged in the supress method called before the current merge
        self.points = []
        self.divided = False

        for point in self.NE.points:
            self.points.append(point)
        for point in self.NW.points:
            self.points.append(point)
        for point in self.SE.points:
            self.points.append(point)
        for point in self.SW.points:
            self.points.append(point)

        self.NE = None
        self.NW = None
        self.SE = None
        self.SW = None
    
    def insert(self, point):
        """
        Careful : insertion of more superposed points than the maximum capacity of a single quadtree might lead to errors.
        """
        if not self.contains(point):
            return
        
        self.representative = point
        if self.divided:
            self.NE.insert(point)
            self.NW.insert(point)
            self.SE.insert(point)
            self.SW.insert(point)
        else:
            self.points.append(point)
            if len(self.points) > self.capacity:
                self.subdivide()
        
        self.empty = False

    # TODO : This method doesn't work !!!
    # We often want to supress points that moved outside of their previous square.
    def supress(self, point):
        if not self.contains(point):
            return
        
        if point == self.representative :
            self.representative = None

        if self.divided:
            self.NE.supress(point)
            self.NW.supress(point)
            self.SE.supress(point)
            self.SW.supress(point)

            self.representative = self.NE.representative or self.NW.representative or self.SE.representative or self.SW.representative

            self.merge()
            if self.total_points() == 0:
                self.empty = True
        else:
            self.points.remove(point)
            if len(self.points) > 0:
                self.representative = self.points[0]
            else:
                self.empty = True
    
    def find_other_representative_than(self, point):
        if self.representative != point:
            return
        if not self.divided:
            if len(self.points) > 1:
                candidates = self.points.copy()
                # point is in candidates because it is the representative
                candidates.remove(point)
                self.representative = min(candidates, key=lambda p: sqr_distance(p.vect, point.vect))
        else:
            self.NE.find_other_representative_than(point)
            self.NW.find_other_representative_than(point)
            self.SE.find_other_representative_than(point)
            self.SW.find_other_representative_than(point)

            r1, r2, r3, r4 = self.NE.representative, self.NW.representative, self.SE.representative, self.SW.representative
            candidates = []
            for r in [r1, r2, r3, r4]:
                if r is not None and r != point:
                    candidates.append(r)
                    
            self.representative = min(candidates, key=lambda p: sqr_distance(p.vect, point.vect))
    
    def closest_naive(self, point, radius):
        """
        This has a worst case complexity of O(n*log(n)) for one query. It is worse thant the naive method of going through the list of all points.
        However, the worst case complexity assumes that all points are within the same radius, so all points are mutually in collision.
        Thus, in a setting with collisions, it is extremely unlikely that more than 6 points are close enough, and in practice, this
        method is faster than the mathematically optimal method below.
        """
        candidates = self.query_circle(point.vect, radius)
        if point in candidates:
            candidates.remove(point)
        
        if len(candidates) == 0:
            return None

        return min(candidates, key=lambda p: sqr_distance(p.vect, point.vect))

    def closest_breadth(self, point):
        """
        Return the closest point to the given point.

        Caution : this method only works if the maximum capacity of a single quadtree is 1, for an unknown reason.

        Algorithm :
            Maintain a list of interesting squares
            sqrs_0 is the root of the quadtree
            sqrs_{i+1} are the squares whose distance to the point is less than the current minimum distance among the children of sqrs_i.
            The current minimum distance is the distance between the point and the best representative yet.
            Returning the best representative gives the closest point.
        """
        if not self.divided:
            # This case is juste for trees reduced to a single leaf. Plunging into the tree will exclude empty squares.
            candidates = self.points.copy()
            if point in self.points:
                candidates.remove(point)
            if len(self.points) == 0:
                return None
            return min(self.points, key=lambda p: sqr_distance(p.vect, point.vect))
        
        sqrs = [self]
        self.find_other_representative_than(point)

        best = self.representative
        
        while sqrs != []:
            best = min([sqr.representative for sqr in sqrs] + [best],
                           key=lambda p: sqr_distance(p.vect, point.vect)
                          )
            new_sqrs = []
            for sqr in sqrs:
                if not sqr.divided:
                    # In this case, the closest point in sqr is already accounted for in best_rep and we can skip to the next square.
                    continue
                
                for sub_sqr in [sqr.NE, sqr.NW, sqr.SE, sqr.SW]:
                    if (not sub_sqr.empty) \
                        and (dist_to_rect_squared(sub_sqr, point.vect) < sqr_distance(best.vect, point.vect)):
                        
                        sub_sqr.find_other_representative_than(point)
                        if sub_sqr.representative == point:
                            # in this case, the only point in this square is the point we are looking for, so we can skip to the next square
                            continue

                        new_sqrs.append(sub_sqr)
            sqrs = new_sqrs
        return best

    def closest_depth(self, point, best=None, order_regions=True):
        """
        Return the closest point to the given point.

        Algorithm :
            Search in depth for the best point, and discard branches that are further than the current best.
            If a branch has a minimal distance to the point smaller than the current best, it is not impossible that it contains a point closer than the current best.
            Ordering the regions by proximity to the point makes the search much more efficient and allows to discard more branches sooner.
        """
        if self.divided:
            regions = sorted([self.NE, self.NW, self.SE, self.SW], key=lambda r: dist_to_rect_squared(r, point.vect)) \
                        if order_regions \
                        else [self.NE, self.NW, self.SE, self.SW]
            for region in regions:
                d = dist_to_rect_squared(region, point.vect)
                if best is None or d < sqr_distance(best.vect, point.vect):
                    best = region.closest_depth(point, best, order_regions)
        else:
            for other_point in self.points:
                if other_point == point:
                    continue
                d = sqr_distance(other_point.vect, point.vect)
                if best is None or d < sqr_distance(best.vect, point.vect):
                    best = other_point
        
        return best

    def query_rect(self, top_left, height, width):
        """
        Return all points in the given rectangle.
        """
        points = []
        if not self.intersects(top_left, height, width):
            return points

        if not self.divided:
            for point in self.points:
                if point.vect[0] >= top_left[0] and point.vect[0] < top_left[0] + width and \
                   point.vect[1] >= top_left[1] and point.vect[1] < top_left[1] + height:
                    points.append(point)
        else:
            points += self.NE.query_rect(top_left, height, width)
            points += self.NW.query_rect(top_left, height, width)
            points += self.SE.query_rect(top_left, height, width)
            points += self.SW.query_rect(top_left, height, width)

        return points

    def intersects(self, top_left, height, width):
        """
        Return true if the given rectangle intersects this quadtree.
        """
        x, y = self.top_left
        return not (x > top_left[0] + width or x + self.width < top_left[0] or y > top_left[1] + height or y + self.height < top_left[1])
    
    def query_circle(self, center, radius):
        """
        Return all points in the given circle.
        """
        points = []
        if not self.intersects_circle(center, radius):
            return points

        if not self.divided:
            for point in self.points:
                if sqr_distance(point.vect, center) <= radius**2:
                    points.append(point)
        else:
            points += self.NE.query_circle(center, radius)
            points += self.NW.query_circle(center, radius)
            points += self.SE.query_circle(center, radius)
            points += self.SW.query_circle(center, radius)

        return points
    
    def intersects_circle(self, center, radius):
        """
        Return true if the given circle intersects this quadtree.
        May also return true if the circle does not intersect this quadtree, we in fact check if the square containing the circle intersects the quadtree.
        """
        x, y = self.top_left
        return not (x > center[0] + radius or x + self.width < center[0] - radius or y > center[1] + radius or y + self.height < center[1] - radius)

    def clear(self):
        """
        Clear this quadtree.
        """
        self.empty = True
        self.representative = None
        self.points = []
        self.divided = False
        self.NE = None
        self.NW = None
        self.SE = None
        self.SW = None


class OneCapacityQuadtree:
    def __init__(self, top_left, height, width):
        self.top_left = top_left
        self.height = height
        self.width = width

        # If false, this node is a leaf. Otherwise, it has four children.
        self.divided = False

        self.point = None
        self.representative = None
        self.empty = True

        self.NE = None
        self.NW = None
        self.SE = None
        self.SW = None

    def depth(self):
        """
        Return the depth of this quadtree.
        """
        if not self.divided:
            return 1
        else:
            return 1 + max(self.NE.depth(), self.NW.depth(), self.SE.depth(), self.SW.depth())
    
    def total_points(self):
        """
        Return the total number of points in this quadtree.
        """
        if not self.divided:
            return 0 if self.empty else 1
        else:
            return (self.NE.total_points() + self.NW.total_points() +
                    self.SE.total_points() + self.SW.total_points())
        
    def total_leafs(self):
        """
        Return the total number of leafs in this quadtree.
        """
        if not self.divided:
            return 1
        else:
            return (self.NE.total_leafs() + self.NW.total_leafs() +
                    self.SE.total_leafs() + self.SW.total_leafs())
    
    def contains(self, point):
        """
        Return true if this quadtree contains the given point.
        """
        x, y = self.top_left
        px, py = point.vect
        
        return (px >= x and px < x + self.width and
                py >= y and py < y + self.height)

    def subdivide(self):
        """
        Subdivide this quadtree into four subnodes.
        """
        x, y = self.top_left
        h = self.height / 2
        w = self.width / 2

        self.NE = OneCapacityQuadtree((x + w, y), h, w)
        self.NW = OneCapacityQuadtree((x, y), h, w)
        self.SE = OneCapacityQuadtree((x + w, y + h), h, w)
        self.SW = OneCapacityQuadtree((x, y + h), h, w)

        if self.point is not None:
            self.NE.insert(self.point)
            self.NW.insert(self.point)
            self.SE.insert(self.point)
            self.SW.insert(self.point)

            self.point = None

        self.divided = True
    
    def merge(self):
        """
        Try to merge the children of this quadtree.
        """
        if not self.divided:
            return
        #counting all points every time is very costly, so we count only when all four children are leaves
        if self.NE.divided or self.NW.divided or self.SE.divided or self.SW.divided:
            return
        if self.total_points() > 1:
            return
        
        # theoretically, there is no need to recursively merge children as they are merged in the supress method called before the current merge
        self.point = self.NE.point or self.NW.point or self.SE.point or self.SW.point
        self.divided = False

        self.NE = None
        self.NW = None
        self.SE = None
        self.SW = None

    def insert_child(self, point):
        self.NE.insert(point)
        self.NW.insert(point)
        self.SE.insert(point)
        self.SW.insert(point)
    
    def insert(self, point):
        """
        Careful : insertion of more superposed points than the maximum capacity of a single quadtree might lead to errors.
        """
        if not self.contains(point):
            return
        
        self.representative = point
        if self.divided:
            self.insert_child(point)
        else:
            if self.point is None:
                self.point = point
            else:
                self.subdivide()
                self.insert_child(point)
                self.point = None
        
        self.empty = False

    # TODO : This method doesn't work !!! If the points moved, they might not be in the same square anymore.
    # We often want to supress points that moved outside of their previous square.
    def supress(self, point):
        if not self.contains(point):
            return
        
        if point == self.representative :
            self.representative = None

        if self.divided:
            self.NE.supress(point)
            self.NW.supress(point)
            self.SE.supress(point)
            self.SW.supress(point)

            self.representative = self.NE.representative or self.NW.representative or self.SE.representative or self.SW.representative

            self.merge()
            if self.total_points() == 0:
                self.empty = True
        else:
            if self.point == point:
                self.point = None
                self.empty = True
    
    def find_other_representative_than(self, point):
        if self.representative != point:
            return
        if not self.divided:
            return
        else:
            self.NE.find_other_representative_than(point)
            self.NW.find_other_representative_than(point)
            self.SE.find_other_representative_than(point)
            self.SW.find_other_representative_than(point)

            r1, r2, r3, r4 = self.NE.representative, self.NW.representative, self.SE.representative, self.SW.representative
            mindist = float('inf')
            for r in [r1, r2, r3, r4]:
                if r is not None and r != point:
                    d = sqr_distance(r.vect, point.vect)
                    if d < mindist:
                        mindist = d
                        self.representative = r
    
    def closest_naive(self, point, radius):
        """
        This has a worst case complexity of O(n*log(n)) for one query. It is worse thant the naive method of going through the list of all points.
        However, the worst case complexity assumes that all points are within the same radius, so all points are mutually in collision.
        Thus, in a setting with collisions, it is extremely unlikely that more than 6 points are close enough, and in practice, this
        method is faster than the mathematically optimal method below.
        """
        candidates = self.query_circle(point.vect, radius)
        if point in candidates:
            candidates.remove(point)
        
        if len(candidates) == 0:
            return None

        return min(candidates, key=lambda p: sqr_distance(p.vect, point.vect))

    def closest_breadth(self, point):
        """
        Return the closest point to the given point.

        Caution : this method only works if the maximum capacity of a single quadtree is 1, for an unknown reason.

        Algorithm :
            Maintain a list of interesting squares
            sqrs_0 is the root of the quadtree
            sqrs_{i+1} are the squares whose distance to the point is less than the current minimum distance among the children of sqrs_i.
            The current minimum distance is the distance between the point and the best representative yet.
            Returning the best representative gives the closest point.
        """
        if not self.divided:
            # This case is juste for trees reduced to a single leaf. Plunging into the tree will exclude empty squares.
            if self.empty or self.point == point:
                return None
            return self.point
        
        sqrs = [self]
        self.find_other_representative_than(point)

        best = self.representative
        
        while sqrs != []:
            best = min([sqr.representative for sqr in sqrs] + [best],
                           key=lambda p: sqr_distance(p.vect, point.vect)
                          )
            new_sqrs = []
            for sqr in sqrs:
                if not sqr.divided:
                    # In this case, the closest point in sqr is already accounted for in best_rep and we can skip to the next square.
                    continue
                
                for sub_sqr in [sqr.NE, sqr.NW, sqr.SE, sqr.SW]:
                    if (not sub_sqr.empty) \
                        and (dist_to_rect_squared(sub_sqr, point.vect) < sqr_distance(best.vect, point.vect)):
                        
                        sub_sqr.find_other_representative_than(point)
                        if sub_sqr.representative == point:
                            # in this case, the only point in this square is the point we are looking for, so we can skip to the next square
                            continue
                        new_sqrs.append(sub_sqr)
            sqrs = new_sqrs

        return best

    def closest_depth(self, point, best=None, order_regions=True):
        """
        Return the closest point to the given point.

        Algorithm :
            Search in depth for the best point, and discard branches that are further than the current best.
            If a branch has a minimal distance to the point smaller than the current best, it is not impossible that it contains a point closer than the current best.
            Ordering the regions by proximity to the point makes the search much more efficient and allows to discard more branches sooner.
        """
        if self.divided:
            regions = sorted([self.NE, self.NW, self.SE, self.SW], key=lambda r: dist_to_rect_squared(r, point.vect)) \
                        if order_regions \
                        else [self.NE, self.NW, self.SE, self.SW]
            for region in regions:
                d = dist_to_rect_squared(region, point.vect)
                if best is None or d < sqr_distance(best.vect, point.vect):
                    best = region.closest_depth(point, best, order_regions)
        else:
            if self.point is not None and self.point != point:
                d = sqr_distance(self.point.vect, point.vect)
                if best is None or d < sqr_distance(best.vect, point.vect):
                    best = self.point
        
        return best

    def query_rect(self, top_left, height, width):
        """
        Return all points in the given rectangle.
        """
        points = []
        if not self.intersects(top_left, height, width):
            return points

        if not self.divided:
            if self.point.vect[0] >= top_left[0] and self.point.vect[0] < top_left[0] + width and \
               self.point.vect[1] >= top_left[1] and self.point.vect[1] < top_left[1] + height:
                points.append(self.point)
        else:
            points += self.NE.query_rect(top_left, height, width)
            points += self.NW.query_rect(top_left, height, width)
            points += self.SE.query_rect(top_left, height, width)
            points += self.SW.query_rect(top_left, height, width)

        return points

    def intersects(self, top_left, height, width):
        """
        Return true if the given rectangle intersects this quadtree.
        """
        x, y = self.top_left
        return not (x > top_left[0] + width or x + self.width < top_left[0] or y > top_left[1] + height or y + self.height < top_left[1])
    
    def query_circle(self, center, radius):
        """
        Return all points in the given circle.
        """
        points = []
        if not self.intersects_circle(center, radius):
            return points
        if self.empty:
            return points
        if not self.divided:
            if sqr_distance(self.point.vect, center) <= radius**2:
                points.append(self.point)
        else:
            points += self.NE.query_circle(center, radius)
            points += self.NW.query_circle(center, radius)
            points += self.SE.query_circle(center, radius)
            points += self.SW.query_circle(center, radius)

        return points
    
    def intersects_circle(self, center, radius):
        """
        Return true if the given circle intersects this quadtree.
        May also return true if the circle does not intersect this quadtree, we in fact check if the square containing the circle intersects the quadtree.
        """
        x, y = self.top_left
        return not (x > center[0] + radius or x + self.width < center[0] - radius or y > center[1] + radius or y + self.height < center[1] - radius)

    def clear(self):
        """
        Clear this quadtree.
        """
        self.empty = True
        self.representative = None
        self.point = None
        self.divided = False
        self.NE = None
        self.NW = None
        self.SE = None
        self.SW = None