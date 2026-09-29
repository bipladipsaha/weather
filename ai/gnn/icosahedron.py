import numpy as np
from scipy.spatial import KDTree

class IcosahedronMesh:
    """
    Generates a genuine icosahedral spherical mesh through recursive subdivision.
    """
    def __init__(self, subdivision_level=0):
        self.level = subdivision_level
        self.vertices, self.faces = self._generate_base_icosahedron()
        
        for _ in range(self.level):
            self._subdivide()
            
        # Ensure vertices are normalized to unit sphere
        self._normalize_vertices()
        
        # Calculate edges and remove duplicates
        self._build_topology()
        
    def _generate_base_icosahedron(self):
        """
        Creates the 12 vertices and 20 faces of a regular icosahedron.
        """
        t = (1.0 + np.sqrt(5.0)) / 2.0
        
        vertices = np.array([
            [-1,  t,  0], [ 1,  t,  0], [-1, -t,  0], [ 1, -t,  0],
            [ 0, -1,  t], [ 0,  1,  t], [ 0, -1, -t], [ 0,  1, -t],
            [ t,  0, -1], [ t,  0,  1], [-t,  0, -1], [-t,  0,  1]
        ], dtype=np.float32)
        
        # Normalize vertices
        norms = np.linalg.norm(vertices, axis=1, keepdims=True)
        vertices = vertices / norms
        
        faces = np.array([
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
        ], dtype=np.int32)
        
        return vertices, faces

    def _subdivide(self):
        """
        Recursively subdivides each triangular face into 4 smaller triangles.
        """
        new_faces = []
        midpoint_cache = {}
        
        def get_midpoint(v1_idx, v2_idx):
            # Use sorted tuple as key to share midpoints between adjacent faces
            key = tuple(sorted((v1_idx, v2_idx)))
            if key in midpoint_cache:
                return midpoint_cache[key]
                
            mid = (self.vertices[v1_idx] + self.vertices[v2_idx]) / 2.0
            new_idx = len(self.vertices)
            # Append is slow for numpy arrays, but fine for low subdivision levels
            self.vertices = np.vstack([self.vertices, mid])
            midpoint_cache[key] = new_idx
            return new_idx
            
        for face in self.faces:
            v0, v1, v2 = face
            
            a = get_midpoint(v0, v1)
            b = get_midpoint(v1, v2)
            c = get_midpoint(v2, v0)
            
            new_faces.extend([
                [v0, a, c],
                [v1, b, a],
                [v2, c, b],
                [a, b, c]
            ])
            
        self.faces = np.array(new_faces, dtype=np.int32)

    def _normalize_vertices(self):
        """
        Projects all vertices onto the unit sphere.
        """
        norms = np.linalg.norm(self.vertices, axis=1, keepdims=True)
        self.vertices = self.vertices / norms
        
    def _build_topology(self):
        """
        Removes any potential duplicate vertices and builds the edge adjacency list.
        """
        # Round coordinates to remove floating point discrepancies, then find unique
        _, unique_indices, inverse_indices = np.unique(
            np.round(self.vertices, decimals=5), axis=0, return_index=True, return_inverse=True
        )
        
        self.vertices = self.vertices[unique_indices]
        self.faces = inverse_indices[self.faces]
        
        # Build edges (undirected)
        edges = set()
        for face in self.faces:
            edges.add(tuple(sorted((face[0], face[1]))))
            edges.add(tuple(sorted((face[1], face[2]))))
            edges.add(tuple(sorted((face[2], face[0]))))
            
        self.edges = np.array(list(edges), dtype=np.int32)
        
        # Vertex-to-vertex adjacency list
        self.adjacency = {i: [] for i in range(len(self.vertices))}
        for u, v in self.edges:
            self.adjacency[u].append(v)
            self.adjacency[v].append(u)
            
    def get_stats(self):
        """
        Returns mesh statistics and validates Euler characteristic.
        """
        V = len(self.vertices)
        E = len(self.edges)
        F = len(self.faces)
        euler = V - E + F
        
        degrees = [len(adj) for adj in self.adjacency.values()]
        
        return {
            "level": self.level,
            "vertices": V,
            "edges": E,
            "faces": F,
            "euler_characteristic": euler,
            "euler_valid": euler == 2,
            "mean_degree": np.mean(degrees),
            "min_degree": np.min(degrees),
            "max_degree": np.max(degrees)
        }
