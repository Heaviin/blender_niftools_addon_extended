"""Release override for nifgen's meshoptimizer triangle strip wrapper."""

import ctypes
import os.path

import numpy as np


meshopt = ctypes.CDLL(os.path.join(os.path.dirname(__file__), "meshoptimizer.dll"))


def stripify(triangles, vertex_count):
    """Stripify triangles, optimizing for the vertex cache."""
    points = [vertex for triangle in triangles for vertex in triangle]
    point_count = len(points)
    cached_points = cache_vertices(points, point_count, vertex_count)
    cached_point_count = len(cached_points)
    strip_bound = meshopt_stripify_bound(cached_point_count)
    strips = meshopt_stripify(points, cached_point_count, strip_bound, vertex_count)
    return [strips.tolist()]


def cache_vertices(points, point_count, vertex_count):
    return meshopt_optimize_vertex_cache_strip(points, point_count, vertex_count).tolist()


meshopt.meshopt_optimizeVertexCacheStrip.restype = None
meshopt.meshopt_optimizeVertexCacheStrip.argtypes = [
    ctypes.POINTER(ctypes.c_uint),
    ctypes.POINTER(ctypes.c_uint),
    ctypes.c_size_t,
    ctypes.c_size_t,
]


def meshopt_optimize_vertex_cache_strip(points, point_count, vertex_count):
    # meshoptimizer uses uint32_t index buffers. np.uint is platform-sized
    # (64-bit on 64-bit Windows) and corrupts indices passed through c_uint.
    output_array = np.zeros(point_count, dtype=np.uint32)
    points_array = np.array(points, dtype=np.uint32).flatten()

    meshopt.meshopt_optimizeVertexCacheStrip(
        output_array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint)),
        points_array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint)),
        ctypes.c_size_t(point_count),
        ctypes.c_size_t(vertex_count),
    )
    return output_array


meshopt.meshopt_stripify.restype = ctypes.c_size_t
meshopt.meshopt_stripify.argtypes = [
    ctypes.POINTER(ctypes.c_uint),
    ctypes.POINTER(ctypes.c_uint),
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_uint,
]


def meshopt_stripify(points, points_count, strip_bound, vertex_count):
    output_array = np.zeros(strip_bound, dtype=np.uint32)
    points_array = np.array(points, dtype=np.uint32).flatten()

    strip_size = meshopt.meshopt_stripify(
        output_array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint)),
        points_array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint)),
        ctypes.c_size_t(points_count),
        ctypes.c_size_t(vertex_count),
        ctypes.c_uint(0),
    )
    return output_array[:strip_size]


meshopt.meshopt_stripifyBound.restype = ctypes.c_size_t
meshopt.meshopt_stripifyBound.argtypes = [ctypes.c_size_t]


def meshopt_stripify_bound(points_count):
    return meshopt.meshopt_stripifyBound(ctypes.c_size_t(points_count))
