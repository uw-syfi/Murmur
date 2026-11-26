"""Centroid linkage and speaker assignment for pyannote 3.1 on MLX."""

import math

import mlx.core as mx

THRESHOLD = 0.7045654963945799


def selected(mask):
    """MLX indices for a dynamic-length boolean selection."""
    count = int(mask.sum().item())
    return mx.argsort(-mask.astype(mx.int32))[:count]


def unit(x):
    return x * mx.rsqrt(mx.maximum(mx.square(x).sum(axis=-1, keepdims=True), 1e-20))


def linkage(points):
    """Build the centroid dendrogram with bounded distance-matrix storage.

    Cache each live centroid's closest neighbour. After a merge only rows
    pointing at an obsolete centroid need a new full nearest-neighbour search.
    Other rows need just one comparison against the new centroid.
    """
    n = len(points)
    points = points[:]  # Updates replace live centroids while preserving the input embeddings.
    ids = mx.arange(n)
    alive = mx.ones((n,), dtype=mx.bool_)
    sizes = mx.ones(2 * n - 1, dtype=mx.int32)
    nodes = mx.arange(n)
    children = mx.zeros((n - 1, 2), dtype=mx.int32)
    heights = mx.zeros(n - 1)
    subtree_height = mx.zeros(2 * n - 1)
    nearest = mx.zeros((n,), dtype=mx.int32)
    best = mx.full((n,), math.inf)

    def refresh(rows):
        nonlocal nearest, best
        norms = mx.square(points).sum(axis=1)
        for start in range(0, len(rows), 128):
            r = rows[start:start + 128]
            square = norms[r, None] + norms - 2 * (points[r] @ points.T)
            distances = mx.where(alive & (ids != r[:, None]), square, math.inf)
            j = distances.argmin(axis=1).astype(mx.int32)
            nearest[r] = j
            best[r] = distances[mx.arange(len(r)), j]
            mx.eval(nearest, best)

    refresh(ids)
    for step in range(n - 1):
        i = best.argmin().astype(mx.int32)
        j = nearest[i]
        left, right = nodes[i], nodes[j]
        parent = n + step
        children[step] = mx.stack([mx.minimum(left, right), mx.maximum(left, right)])
        heights[step] = mx.sqrt(mx.maximum(best[i], 0))
        subtree_height[parent] = mx.maximum(heights[step], mx.maximum(subtree_height[left], subtree_height[right]))
        sizes[parent] = sizes[left] + sizes[right]
        points[i] = (sizes[left] * points[i] + sizes[right] * points[j]) / sizes[parent]
        nodes[i] = parent
        mx.eval(children, heights, subtree_height, sizes, nodes, points)
        alive[j] = False
        if step == n - 2:
            break
        stale = alive & ((nearest == i) | (nearest == j) | (ids == i))
        to_new = mx.square(points - points[i]).sum(axis=1)
        improved = alive & (ids != i) & (to_new < best)
        nearest = mx.where(improved, i, nearest)
        best = mx.where(alive, mx.minimum(to_new, best), math.inf)
        refresh(selected(stale))
    return children, heights, sizes, subtree_height


def cut_tree(children, heights, sizes, subtree_height, minimum, requested):
    """Choose the pyannote threshold/count cut and label leaves in tree order."""
    n = len(children) + 1
    min_size = min(12, max(1, round(n * 0.1)))
    ids = mx.arange(2 * n - 1)
    parent = mx.arange(2 * n - 1)
    parent[children.reshape(-1)] = mx.repeat(ids[n:], 2)
    root = ids == 2 * n - 2
    # A centroid merge can be lower than either child's merge height.
    below = subtree_height <= THRESHOLD
    large = sizes >= min_size
    threshold_count = (below & (~below[parent] | root) & large).sum()

    # Given an out-of-bounds speaker count, choose the iteration giving
    # the closest large-cluster count, breaking ties by threshold distance.
    change = large[n:].astype(mx.int32) - large[children].sum(axis=1).astype(mx.int32)
    count = (n if min_size == 1 else 0) + mx.cumsum(change)
    target = requested if requested is not None else mx.maximum(minimum or 1, threshold_count)
    target = mx.minimum(n, target)
    error = mx.where(large[n:], mx.abs(count - target), n + 1)
    iteration = mx.argmin(mx.where(error == error.min(), mx.abs(heights - THRESHOLD), math.inf)).astype(mx.int32)
    below = mx.where(threshold_count == target, below, ids <= n + iteration)

    # Pointer doubling finds each leaf's selected ancestor. The right
    # child's offset in depth-first order is the size of its left sibling.
    ancestor = parent
    group = mx.where(below[parent], parent, ids)
    offset = mx.zeros_like(ids)
    offset[children[:, 1]] = sizes[children[:, 0]]
    for _ in range((2 * n - 1).bit_length()):
        offset = offset + offset[ancestor]
        ancestor = ancestor[ancestor]
        group = group[group]
    roots = selected(below & (~below[parent] | root))
    roots = roots[mx.argsort(offset[roots])]
    labels = mx.zeros_like(ids)
    labels[roots] = mx.arange(len(roots))
    return labels[group[:n]], sizes[roots], min_size


def assign(embeddings, activity, minimum=None, requested=None):
    clean = activity * (activity.sum(axis=-1, keepdims=True) == 1)
    valid = (clean.sum(axis=1) >= 0.2 * activity.shape[1]) & mx.isfinite(embeddings).all(axis=-1)
    rows = selected(valid.reshape(-1))
    if len(rows) < 2 or requested == 1:
        return mx.zeros(embeddings.shape[:2], dtype=mx.int32), 1
    train = embeddings.reshape(-1, embeddings.shape[-1])[rows]
    normalized = unit(train)
    tree = linkage(normalized)
    labels, sizes, min_size = cut_tree(*tree, minimum, requested)
    large = selected(sizes >= min_size)
    if len(large) < 2:
        return mx.zeros(embeddings.shape[:2], dtype=mx.int32), 1

    cluster_sizes = sizes[:, None]
    sums = mx.zeros((len(sizes), train.shape[-1])).at[labels].add(normalized)
    centers = unit(sums / cluster_sizes)
    remap = (centers @ centers[large].T).argmax(axis=-1)
    remap[large] = mx.arange(len(large))
    labels = remap[labels]
    # pyannote assigns all local speakers by cosine similarity against the
    # mean original (unnormalized) training embedding in each surviving group.
    centers = mx.zeros((len(large), train.shape[-1])).at[labels].add(train)
    return (embeddings @ unit(centers).T).argmax(axis=-1), len(large)
