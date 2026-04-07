import jax.numpy as jnp


VALID_NEIGHBOR_LIST_MODES = ("none", "single", "many-body")


def validate_neighbor_list_mode(mode):
    if mode not in VALID_NEIGHBOR_LIST_MODES:
        available = ", ".join(VALID_NEIGHBOR_LIST_MODES)
        raise ValueError(f"Unknown neighbor-list mode '{mode}'. Available: {available}.")
    return mode


def build_neighbor_list(
    positions,
    box=None,
    cutoff=None,
    skin=0.0,
    mode="single",
    pbc=False,
    padded=True,
    max_neighbors=None,
):
    """Return topology-only interaction indices for the requested neighbor-list mode."""
    mode = validate_neighbor_list_mode(mode)

    if mode == "none":
        return None
    if cutoff is None:
        raise ValueError("cutoff must be provided when building a neighbor list.")
    if mode == "single":
        return _build_single_neighbor_list(
            positions,
            box=box,
            cutoff=cutoff,
            skin=skin,
            pbc=pbc,
            padded=padded,
            max_neighbors=max_neighbors,
        )
    if mode == "many-body":
        return _build_many_body_neighbor_list(
            positions,
            box=box,
            cutoff=cutoff,
            skin=skin,
            pbc=pbc,
            padded=padded,
            max_neighbors=max_neighbors,
        )

    raise ValueError(f"Unsupported neighbor-list mode '{mode}'.")


def _build_single_neighbor_list(
    positions,
    box=None,
    cutoff=None,
    skin=0.0,
    pbc=False,
    padded=True,
    max_neighbors=None,
):
    """Build a one-sided neighbor topology where each pair appears once."""
    search_cutoff = cutoff + skin
    neighbor_mask = _compute_neighbor_mask(
        positions,
        box=box,
        search_cutoff=search_cutoff,
        pbc=pbc,
    )
    neighbor_mask = jnp.triu(neighbor_mask, k=1)

    if not padded:
        return _mask_to_pair_indices(neighbor_mask)

    return _pad_neighbor_mask(neighbor_mask, max_neighbors=max_neighbors)


def _build_many_body_neighbor_list(
    positions,
    box=None,
    cutoff=None,
    skin=0.0,
    pbc=False,
    padded=True,
    max_neighbors=None,
):
    """Build a symmetric neighbor topology where each particle sees all neighbors."""
    search_cutoff = cutoff + skin
    neighbor_mask = _compute_neighbor_mask(
        positions,
        box=box,
        search_cutoff=search_cutoff,
        pbc=pbc,
    )

    if not padded:
        return _mask_to_pair_indices(neighbor_mask)

    return _pad_neighbor_mask(neighbor_mask, max_neighbors=max_neighbors)


def _compute_neighbor_mask(positions, box=None, search_cutoff=None, pbc=False):
    """Return a dense boolean adjacency matrix for the current geometry."""
    if search_cutoff is None:
        raise ValueError("search_cutoff must be provided to build a neighbor list.")

    dr = positions[:, None, :] - positions[None, :, :]
    if pbc:
        dr = minimum_image(dr, box=box)

    dist = jnp.linalg.norm(dr, axis=-1)
    self_mask = ~jnp.eye(positions.shape[0], dtype=bool)
    return (dist < search_cutoff) & self_mask


def _mask_to_pair_indices(neighbor_mask):
    """Convert a dense boolean adjacency matrix into a flat pair-index array."""
    pair_indices = jnp.argwhere(neighbor_mask)
    return pair_indices.astype(jnp.int32)


def _pad_neighbor_mask(neighbor_mask, max_neighbors=None, pad_value=-1):
    """Convert a dense boolean neighbor mask into padded integer neighbor indices."""
    counts = jnp.sum(neighbor_mask, axis=1)
    if max_neighbors is None:
        inferred_max_neighbors = int(jnp.max(counts))
        pad_width = inferred_max_neighbors
    else:
        pad_width = int(max_neighbors)
        try:
            inferred_max_neighbors = int(jnp.max(counts))
        except TypeError:
            inferred_max_neighbors = None

        if inferred_max_neighbors is not None and pad_width < inferred_max_neighbors:
            raise ValueError(
                f"max_neighbors={pad_width} is smaller than the required width "
                f"{inferred_max_neighbors} for the current neighbor list."
            )

    if pad_width == 0:
        n_particles = neighbor_mask.shape[0]
        empty_neighbors = jnp.full((n_particles, 0), pad_value, dtype=jnp.int32)
        empty_mask = jnp.zeros((n_particles, 0), dtype=bool)
        return empty_neighbors, empty_mask

    sorted_indices = jnp.argsort(~neighbor_mask, axis=1)
    available_width = sorted_indices.shape[1]
    take_width = min(pad_width, available_width)
    candidate_neighbors = sorted_indices[:, :take_width]
    gathered_mask = jnp.take_along_axis(neighbor_mask, candidate_neighbors, axis=1)
    count_mask = jnp.arange(take_width)[None, :] < counts[:, None]
    padded_mask = gathered_mask & count_mask
    padded_neighbors = jnp.where(
        padded_mask,
        candidate_neighbors,
        jnp.full(candidate_neighbors.shape, pad_value, dtype=jnp.int32),
    )

    if take_width < pad_width:
        pad_cols = pad_width - take_width
        padded_neighbors = jnp.pad(
            padded_neighbors,
            ((0, 0), (0, pad_cols)),
            constant_values=pad_value,
        )
        padded_mask = jnp.pad(
            padded_mask,
            ((0, 0), (0, pad_cols)),
            constant_values=False,
        )

    return padded_neighbors, padded_mask



def minimum_image(displacements, box=None):
    """Apply a minimum-image convention to displacement vectors."""
    if box is None:
        return displacements

    box = jnp.asarray(box, dtype=displacements.dtype)
    if box.ndim == 1:
        return displacements - box * jnp.round(displacements / box)
    if box.shape == (3, 3):
        inv_box = jnp.linalg.inv(box)
        fractional = jnp.einsum("...i,ij->...j", displacements, inv_box)
        fractional = fractional - jnp.round(fractional)
        return jnp.einsum("...i,ij->...j", fractional, box)

    raise ValueError(f"Expected box shape (3,) or (3, 3), got {box.shape}.")


def pair_distances(positions, neighbor_list, box=None):
    """Return displacement vectors, scalar distances, and a padding mask."""
    if isinstance(neighbor_list, tuple):
        neighbor_indices, mask = neighbor_list
    else:
        neighbor_indices = neighbor_list
        mask = neighbor_indices >= 0

    safe_neighbor_list = jnp.where(mask, neighbor_indices, 0)
    r_j = positions[safe_neighbor_list]
    raw_dr = minimum_image(r_j - positions[:, None, :], box)
    filler = jnp.broadcast_to(
        jnp.array([1.0, 0.0, 0.0], dtype=raw_dr.dtype),
        raw_dr.shape,
    )
    safe_dr = jnp.where(mask[..., None], raw_dr, filler)
    dist = jnp.linalg.norm(safe_dr, axis=-1)
    dr = jnp.where(mask[..., None], raw_dr, 0.0)
    dist = jnp.where(mask, dist, 0.0)
    return dr, dist, mask
