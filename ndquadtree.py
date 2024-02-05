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

def distance_sqr(p1, p2):
    return np.sum((p1 - p2)**2, axis=-1)

def mahalanobis_distance_sqr(p1, p2, cov, inv=False):
    diff = p1 - p2
    if inv:
        return np.sum(diff @ cov @ diff, axis=-1)
    return np.sum(diff @ np.linalg.inv(cov) @ diff, axis=-1)

def d2r_sqr(rect, point):
    """
    distance to rectangle squared
    """
    clamped = np.minimum(np.maximum(point, rect.v1), rect.v1 + rect.v2)
    return distance_sqr(point, clamped)

def malahanobis_d2r_sqr(rect, point, cov, inv=False):
    """
    distance to rectangle squared
    """
    clamped = np.minimum(np.maximum(point, rect.v1), rect.v1 + rect.v2)
    return mahalanobis_distance_sqr(point, clamped, cov, inv)


class Rectangle:
    def __init__(self, v1, v2):
        """
        v1 is the coordinates of the "top left" corner of the rectangle
        v2 is the width and height of the rectangle, all positive
        """
        self.v1 = v1
        self.v2 = v2
    
    def contains(self, vect):
        """
        Return true if the given point is inside this rectangle.
        """
        return np.all(vect >= self.v1) and np.all(vect < self.v1 + self.v2)
    
    def intersects(self, rect):
        """
        Return true if the given rectangle intersects this rectangle.
        """
        v1 = np.maximum(self.v1, rect.v1)
        v2 = np.minimum(self.v1 + self.v2, rect.v1 + rect.v2)
        return np.all(v1 < v2)
    
    def intersects_circle(self, center, radius, dist_fct=distance_sqr):
        """
        Return true if the given circle intersects this rectangle.
        """
        clamped = np.minimum(np.maximum(center, self.v1), self.v1 + self.v2)
        return dist_fct(center, clamped) < radius**2
    

class NDQuadtree:
    def __init__(self, rect, ndim=None, isotropic=True, cov=None, object_type=None):
        """
        rect : limit rectangle of the quadtree.
        ndim : dimension of the quadtree. If None, it is inferred from the dimension of the rectangle.
        isotropic : if True, the distance function is the euclidean distance. Otherwise, it is the mahalanobis distance given by the covariance matrix.
        cov : covariance matrix of the mahalanobis distance. If None, the identity matrix is used.
        object_type : type of the objects stored in the quadtree. If None, the quadtree stores arrays corresponding to the coordinates.
                      Otherwise, it stores an object that must have a vect attribute, which is a numpy array of size ndim.
        """
        self.rect = rect
        if ndim is None:
            ndim = len(rect.v1)
        assert len(rect.v1) == ndim, "The dimension of the rectangle and the dimension of the quadtree should be the same."
        self.ndim = ndim
        self.object_type = object_type

        self.isotropic = isotropic
        self.cov = cov

        if not self.isotropic:
            if self.cov is None:
                self.cov = np.eye(self.ndim)
            elif np.isscalar(self.cov):
                self.cov = np.eye(self.ndim) * self.cov
            elif self.cov.ndim == 1:
                self.cov = np.diag(self.cov)
            elif self.cov.ndim == 2:
                assert self.cov.shape == (self.ndim, self.ndim), "The covariance matrix should be of size ndim x ndim."
            else:
                raise ValueError("The covariance matrix should be one of the following: None, a scalar, a 1D array or a 2D array.")

            self.cov_inv = np.linalg.inv(self.cov)
        
        if self.isotropic:
            self.dist_fct = distance_sqr
            self.d2r_fct = d2r_sqr
        else:
            self.dist_fct = lambda p1, p2: mahalanobis_distance_sqr(p1, p2, self.cov_inv, inv=True)
            self.d2r_fct = lambda rect, point: malahanobis_d2r_sqr(rect, point, self.cov_inv, inv=True)

        # If false, this node is a leaf. Otherwise, it has 2^ndim children.
        self.empty = True

        self.point = None
        self.representative = None

        self.divided = False
        self.children = [None for _ in range(2**ndim)]

    def depth(self):
        """
        Return the depth of this quadtree.
        """
        if not self.divided:
            return 1
        else:
            return 1 + max([child.depth() for child in self.children])
    
    def total_points(self):
        """
        Return the total number of points in this quadtree.
        """
        if not self.divided:
            return 0 if self.empty else 1
        else:
            return sum([child.total_points() for child in self.children])
        
    def total_leafs(self):
        """
        Return the total number of leafs in this quadtree.
        """
        if not self.divided:
            return 1
        else:
            return sum([child.total_leafs() for child in self.children])
    
    def contains(self, point):
        """
        Return true if this quadtree contains the given point.
        """
        if self.object_type is None:
            return self.rect.contains(point)
        return self.rect.contains(point.vect)

    def subdivide(self):
        """
        Subdivide this quadtree into four subnodes.
        """
        v1, v2 = self.rect.v1, self.rect.v2
        new_v2 = v2 / 2

        for i in range(2**self.ndim):
            new_v1 = v1 + new_v2 * np.array([i//(2**j) % 2 for j in range(self.ndim)])
            self.children[i] = NDQuadtree(Rectangle(new_v1, new_v2), ndim=self.ndim, isotropic=self.isotropic, cov=self.cov, object_type=self.object_type)

        if self.point is not None:
            self.insert_child(self.point)
        self.point = None
        
        self.divided = True

    def insert_child(self, point):
        for child in self.children:
            child.insert(point)
    
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
    
    def closest_naive(self, point, radius):
        """
        This has a worst case complexity of O(n*log(n)) for one query. It is worse thant the naive method of going through the list of all points.
        However, the worst case complexity assumes that all points are within the same radius, so all points are mutually in collision.
        Thus, in a setting with collisions, it is extremely unlikely that more than 6 points are close enough, and in practice, this
        method is faster than the mathematically optimal method below.
        """
        if self.object_type is None:
            candidates = self.query_circle(point, radius)
        else:
            candidates = self.query_circle(point.vect, radius)
            if point in candidates:
                candidates.remove(point)
        
        if len(candidates) == 0:
            return None

        if self.object_type is None:
            return candidates[np.argmin(self.dist_fct(candidates, point))]#TODO : query circle should return an array
        return min(candidates, key=lambda p: self.dist_fct(p.vect, point.vect))

    def closest_depth(self, point, best=None):
        """
        Return the closest point to the given point.

        Algorithm :
            Search in depth for the best point, and discard branches that are further than the current best.
            If a branch has a minimal distance to the point smaller than the current best, it is not impossible that it contains a point closer than the current best.
            Ordering the regions by proximity to the point makes the search much more efficient and allows to discard more branches sooner.
        """
        if self.object_type is None:
            if self.divided:
                regions = sorted(self.children, key=lambda region: self.d2r_fct(region.rect, point))
                for region in regions:
                    d = self.d2r_fct(region.rect, point)
                    # if the closest possible point of the region is further than the current best, we can discard the region
                    if best is None or d < self.dist_fct(best, point):
                        best = region.closest_depth(point, best)
            else:
                if self.point is not None and ((self.point - point)**2).sum() > 1e-15:
                    d = self.dist_fct(self.point, point)
                    if best is None or d < self.dist_fct(best, point):
                        best = self.point
        else :
            if self.divided:
                regions = sorted(self.children, key=lambda region: self.d2r_fct(region.rect, point.vect))
                for region in regions:
                    d = self.d2r_fct(region.rect, point.vect)
                    # if the closest possible point of the region is further than the current best, we can discard the region
                    if best is None or d < self.dist_fct(best.vect, point.vect):
                        best = region.closest_depth(point, best)
            else:
                if self.point is not None and self.point != point:
                    d = self.dist_fct(self.point.vect, point.vect)
                    if best is None or d < self.dist_fct(best.vect, point.vect):
                        best = self.point
        
        return best

    def _query_rect(self, rect):
        """
        Return all points in the given rectangle.
        """
        points = []
        if self.empty or not self.intersects(rect):
            return points

        if not self.divided:
            if rect.contains(self.point.vect if self.object_type is not None else self.point):
                points.append(self.point)
        else:
            for child in self.children:
                points += child._query_rect(rect)

        return points
    
    def query_rect(self, rect):
        """
        Return all points in the given rectangle.
        """
        points = self._query_rect(rect)
        if self.object_type is None:
            return np.array(points)
        return points

    def intersects(self, rect):
        """
        Return true if the given rectangle intersects this quadtree.
        """
        return self.rect.intersects(rect)
        
    def _query_circle(self, center, radius):
        """
        center should be a numpy array of size self.ndim.
        Return all points in the given circle.
        """
        points = []
        if self.empty:
            return points
        if not self.intersects_circle(center, radius):
            return points
        if not self.divided:
            if self.object_type is None:
                if self.dist_fct(self.point, center) <= radius**2 and ((self.point - center)**2).sum() > 1e-15:
                    points.append(self.point)
            else:
                if self.dist_fct(self.point.vect, center) <= radius**2:
                    points.append(self.point)
        else:
            for child in self.children:
                points += child._query_circle(center, radius)

        return points
    
    def query_circle(self, center, radius):
        """
        center should be a numpy array of size self.ndim.
        Return all points in the given circle.
        """
        points = self._query_circle(center, radius)
        if self.object_type is None:
            return np.array(points)
        return points
    
    def intersects_circle(self, center, radius):
        """
        Return true if the given circle intersects this quadtree.
        May also return true if the circle does not intersect this quadtree, we in fact check if the square containing the circle intersects the quadtree.
        """
        return self.rect.intersects_circle(center, radius, dist_fct=self.dist_fct)
        
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