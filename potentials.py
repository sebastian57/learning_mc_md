import jax.numpy as jnp
from functools import partial
from neighbor_list import pair_distances


class HarmonicPotential:
    """Harmonic potential V(r) = 0.5 * k * ||r - r0||^2, per atom."""

    def __init__(self, k=1.0, r0=1.0):
        self.k = k
        self.r0 = r0

    def __call__(self, r):
        return 0.5 * self.k * jnp.sum((r - self.r0) ** 2)


def lennard_jones_potential(
    positions,
    neighbor_list,
    box,
    eps=1.0,
    sigma=1.0,
    r_cut=None,
    shift=False,
):
    dr, dist, mask = pair_distances(positions, neighbor_list, box)

    interaction_mask = mask if r_cut is None else (mask & (dist < r_cut))
    safe_dist = jnp.where(interaction_mask, jnp.maximum(dist, 1e-12), 1.0)

    sr6 = (sigma / safe_dist) ** 6
    sr12 = sr6 ** 2
    pair_energy = 4.0 * eps * (sr12 - sr6)

    if shift and r_cut is not None:
        sr6_cut = (sigma / r_cut) ** 6
        sr12_cut = sr6_cut ** 2
        e_cut = 4.0 * eps * (sr12_cut - sr6_cut)
        pair_energy = pair_energy - e_cut

    pair_energy = jnp.where(interaction_mask, pair_energy, 0.0)

    return 0.5 * jnp.sum(pair_energy)


POTENTIALS = {
    "harmonic": HarmonicPotential,
    "lennard_jones": lennard_jones_potential,
}


def get_potential(name="harmonic", **kwargs):
    """Build a potential by registry name or return the callable unchanged."""
    if callable(name):
        return name
    if name not in POTENTIALS:
        available = ", ".join(sorted(POTENTIALS))
        raise KeyError(f"Unknown potential '{name}'. Available: {available}.")
    potential = POTENTIALS[name]
    return potential(**kwargs) if isinstance(potential, type) else partial(potential, **kwargs)


# Default potential used by md.py's CLI loader when this module is passed in.
potential = HarmonicPotential()
