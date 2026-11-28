"""Bounded centroid-linkage bookkeeping for Metal speaker embeddings."""

import math

import mlx.core as mx
import numpy as np


# One threadgroup per refreshed row. The input contains MLX's original
# squared distances, including negative roundoff. Ties choose the first slot.
_closest = mx.fast.metal_kernel(
    name="murmur_linkage_closest", input_names=["square", "rows", "alive"],
    output_names=["neighbors", "values"],
    source="""
        uint lane = thread_position_in_threadgroup.x;
        uint row = threadgroup_position_in_grid.x;
        uint count = square_shape[1];
        float best = INFINITY;
        uint nearest = count;
        for (uint j = lane; j < count; j += 128) {
            float value = square[row * count + j];
            if (alive[j] && j != uint(rows[row]) &&
                (value < best || (value == best && j < nearest))) {
                best = value;
                nearest = j;
            }
        }
        // A 128-thread group uses 1 KiB of shared memory. This
        // stable reduction is independent of the hardware SIMD width.
        threadgroup float minima[128];
        threadgroup uint indices[128];
        minima[lane] = best;
        indices[lane] = nearest;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = 64; stride > 0; stride >>= 1) {
            if (lane < stride) {
                float other = minima[lane + stride];
                uint index = indices[lane + stride];
                if (other < best || (other == best && index < nearest)) {
                    best = other;
                    nearest = index;
                    minima[lane] = best;
                    indices[lane] = nearest;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lane == 0) {
            neighbors[row] = nearest;
            values[row] = best;
        }
    """,
)


def linkage(points):
    """Keep the merge schedule in shared host memory and its FP32 math on Metal."""
    n = len(points)
    ids = np.arange(n, dtype=np.int32)
    alive = np.ones(n, dtype=np.bool_)
    nearest = np.empty(n, dtype=np.int32)
    best = np.empty(n, dtype=np.float32)
    nodes = ids.copy()
    children = np.empty((n - 1, 2), dtype=np.int32)
    sizes = np.ones(2 * n - 1, dtype=np.int32)
    squared_heights = np.zeros(2 * n - 1, dtype=np.float32)
    subtree_square = np.zeros_like(squared_heights)
    # Inputs belong to the caller; replace rows only in this local view.
    points = points[:]

    def distances(rows):
        # Keep the stock matrix shapes, reduction axes and centroid slot order.
        # MLX can select different BLAS accumulation orders for GEMV and GEMM.
        norms = mx.square(points).sum(axis=1)
        live = mx.array(alive)
        batches = []
        # At most one 128-by-n distance tile (1 MiB at the cap) is
        # pending. Metadata is linear; centroid storage is n-by-d.
        for start in range(0, len(rows), 128):
            if start:
                mx.eval(*batches[-2:])
            indices = mx.array(rows[start:start + 128])
            square = norms[indices, None] + norms - 2 * (points[indices] @ points.T)
            shape = (len(indices),)
            batches.extend(_closest(
                inputs=[square, indices, live], grid=(128 * len(indices), 1, 1),
                threadgroup=(128, 1, 1), output_shapes=[shape, shape],
                output_dtypes=[mx.int32, mx.float32],
            ))
        return batches

    def refresh(rows, outputs):
        for start, neighbors, values in zip(range(0, len(rows), 128), outputs[::2], outputs[1::2]):
            selected = rows[start:start + 128]
            nearest[selected] = np.asarray(neighbors)
            best[selected] = np.asarray(values)

    batches = distances(ids)
    mx.eval(*batches)
    refresh(ids, batches)
    for step in range(n - 1):
        i = best.argmin()
        j = nearest[i]
        left, right = nodes[i], nodes[j]
        parent = n + step
        children[step] = min(left, right), max(left, right)
        squared_heights[parent] = np.maximum(best[i], np.float32(0))
        subtree_square[parent] = max(squared_heights[parent], subtree_square[left], subtree_square[right])
        sizes[parent] = sizes[left] + sizes[right]
        nodes[i] = parent
        alive[j] = False
        if step == n - 2:
            break

        a, b = int(i), int(j)
        points[a] = (float(sizes[left]) * points[a] + float(sizes[right]) * points[b]) / float(sizes[parent])
        stale = np.flatnonzero(alive & ((nearest == i) | (nearest == j) | (ids == i))).astype(np.int32)
        to_new = mx.square(points - points[a]).sum(axis=1)
        batches = distances(stale)
        # One fence per merge. Fresh neighbors use BLAS distances; surviving
        # cached rows improve only on a strictly smaller direct squared norm.
        mx.eval(to_new, *batches)
        direct = np.asarray(to_new)
        improved = alive & (ids != i) & (direct < best)
        nearest[improved] = i
        np.minimum(best, direct, out=best)
        best[~alive] = math.inf
        refresh(stale, batches)

    return (mx.array(children), mx.sqrt(mx.array(squared_heights[n:])), mx.array(sizes),
            mx.sqrt(mx.array(subtree_square)))
