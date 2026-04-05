import argparse
import importlib.util
import json
import time

import numpy as np
import jax
import jax.numpy as jnp
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# Timer
# ---------------------------------------------------------------------------

class Timer:
    """Context manager for timing code blocks.

    Usage
    -----
    with Timer("my label") as t:
        ...
    print(t.elapsed)   # seconds
    """

    def __init__(self, label="", verbose=True):
        self.label = label
        self.verbose = verbose
        self.elapsed = None

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed = time.perf_counter() - self._start
        if self.verbose:
            prefix = f"[{self.label}] " if self.label else ""
            print(f"{prefix}elapsed: {self.elapsed:.4f}s")


# ---------------------------------------------------------------------------
# Potentials
# ---------------------------------------------------------------------------

class HarmonicPotential:
    """Harmonic potential V(r) = 0.5 * k * ||r - r0||^2, per atom.

    Parameters
    ----------
    k  : spring constant
    r0 : equilibrium position (scalar broadcast)
    """

    def __init__(self, k=1.0, r0=1.0):
        self.k = k
        self.r0 = r0

    def __call__(self, r):
        """r : shape (3,) -> scalar energy."""
        return 0.5 * self.k * jnp.sum((r - self.r0) ** 2)


# ---------------------------------------------------------------------------
# MD engine
# ---------------------------------------------------------------------------

class MD:
    """Velocity-Verlet molecular dynamics engine.

    Parameters
    ----------
    r         : array (N, 3)   initial positions
    v         : array (N, 3)   initial velocities
    a         : array (N, 3)   initial accelerations
    potential : callable  potential(r_i) -> scalar, r_i shape (3,)
    dt        : float     timestep
    m         : float     particle mass (uniform)

    After run(), these attributes are available:
        self.r            final positions
        self.v            final velocities
        self.a            final accelerations
        self.trajectory   dict with keys 'positions', 'velocities', 'accelerations',
                          each a numpy array of shape (n_steps, N, 3)
    """

    def __init__(self, r, v, a, potential, dt=0.01, m=1.0):
        self.r = jnp.array(r, dtype=jnp.float32)
        self.v = jnp.array(v, dtype=jnp.float32)
        self.a = jnp.array(a, dtype=jnp.float32)
        self.potential = potential
        self.dt = float(dt)
        self.m = float(m)
        self.trajectory = None

    # ------------------------------------------------------------------
    # Internal step (built once, reused across chunks)
    # ------------------------------------------------------------------

    def _build_step_fn(self):
        force_fn = jax.vmap(lambda r_i: -jax.grad(self.potential)(r_i))
        dt, m = self.dt, self.m

        @jax.jit
        def step(carry, _):
            r, v, a = carry
            r_ = r + v * dt + 0.5 * a * dt ** 2
            a_ = force_fn(r_) / m
            v_ = v + 0.5 * (a + a_) * dt
            return (r_, v_, a_), (r_, v_, a_)

        return step

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, n_steps=100, performance=False, chunk_size=10):
        """Run the simulation for n_steps velocity-Verlet steps.

        Parameters
        ----------
        n_steps     : total number of integration steps
        performance : if True, run as a single lax.scan call with no Python
                      overhead, no synchronisation, and no progress bar —
                      maximum JAX/XLA performance.
                      if False (default), run in chunks with a tqdm progress bar.
        chunk_size  : (ignored when performance=True) steps per lax.scan call;
                      smaller values give finer progress updates at higher overhead.
        """
        step = self._build_step_fn()

        if performance:
            (r_f, v_f, a_f), (r_t, v_t, a_t) = jax.lax.scan(
                step, (self.r, self.v, self.a), None, length=n_steps
            )
            self.r, self.v, self.a = r_f, v_f, a_f
            self.trajectory = {
                "positions":     np.array(r_t),
                "velocities":    np.array(v_t),
                "accelerations": np.array(a_t),
            }
        else:
            n_full, remainder = divmod(n_steps, chunk_size)
            chunk_sizes = [chunk_size] * n_full
            if remainder:
                chunk_sizes.append(remainder)

            carry = (self.r, self.v, self.a)
            all_r, all_v, all_a = [], [], []

            bar = tqdm(
                chunk_sizes,
                desc="MD simulation",
                unit="chunk",
                bar_format="{l_bar}{bar}| chunk {n_fmt}/{total_fmt}  "
                           "[{elapsed}<{remaining}, {rate_fmt}]",
            )

            for cs in bar:
                carry, (r_chunk, v_chunk, a_chunk) = jax.lax.scan(
                    step, carry, None, length=cs
                )
                jax.block_until_ready(carry)
                all_r.append(np.array(r_chunk))
                all_v.append(np.array(v_chunk))
                all_a.append(np.array(a_chunk))

            self.r, self.v, self.a = carry
            self.trajectory = {
                "positions":     np.concatenate(all_r, axis=0),
                "velocities":    np.concatenate(all_v, axis=0),
                "accelerations": np.concatenate(all_a, axis=0),
            }

        return self.trajectory

    # ------------------------------------------------------------------
    # Trajectory persistence
    # ------------------------------------------------------------------

    def save_trajectory(self, path):
        """Save trajectory to a compressed .npz file."""
        if self.trajectory is None:
            raise RuntimeError("No trajectory to save — run the simulation first.")
        np.savez_compressed(path, **self.trajectory)
        print(f"Trajectory saved to {path}.npz")

    @staticmethod
    def load_trajectory(path):
        """Load a trajectory saved with save_trajectory().

        Returns a dict with keys 'positions', 'velocities', 'accelerations'.
        """
        data = np.load(path if path.endswith(".npz") else path + ".npz")
        return {k: data[k] for k in ("positions", "velocities", "accelerations")}


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _parse_array(value, N, rng_key):
    if value == "random":
        return jax.random.normal(rng_key, (N, 3))
    try:
        arr = jnp.array(json.loads(value), dtype=jnp.float32)
    except json.JSONDecodeError:
        raise ValueError(f"Cannot parse '{value}': use 'random' or a JSON array.")
    if arr.shape != (N, 3):
        raise ValueError(f"Expected shape ({N}, 3), got {arr.shape}.")
    return arr


def _load_potential(path):
    spec = importlib.util.spec_from_file_location("custom_potential", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "potential"):
        raise AttributeError(f"{path} must define a callable named 'potential'.")
    return mod.potential


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="JAX Velocity-Verlet MD simulation")

    parser.add_argument("--positions",     default="random")
    parser.add_argument("--velocities",    default="random")
    parser.add_argument("--accelerations", default="random")
    parser.add_argument("--random",        action="store_true",
                        help="Set all of r, v, a to random")
    parser.add_argument("--dt",            type=float, default=0.01)
    parser.add_argument("--m",             type=float, default=1.0)
    parser.add_argument("--N",             type=int,   default=100)
    parser.add_argument("--n_steps",       type=int,   default=100)
    parser.add_argument("--chunk_size",    type=int,   default=10,
                        help="Steps per lax.scan chunk (progress-track mode only)")
    parser.add_argument("--performance",   action="store_true",
                        help="Single lax.scan, no progress bar — maximum performance.")
    parser.add_argument("--potential",     default=None,
                        help="Path to a Python file defining potential(r).")
    parser.add_argument("--save",          default=None,
                        help="Save trajectory to this path (without .npz extension).")

    args = parser.parse_args()

    if args.random:
        args.positions = args.velocities = args.accelerations = "random"

    key = jax.random.PRNGKey(0)
    key_r, key_v, key_a = jax.random.split(key, 3)

    r0 = _parse_array(args.positions,     args.N, key_r)
    v0 = _parse_array(args.velocities,    args.N, key_v)
    a0 = _parse_array(args.accelerations, args.N, key_a)

    potential = _load_potential(args.potential) if args.potential else HarmonicPotential()

    sim = MD(r0, v0, a0, potential, dt=args.dt, m=args.m)

    with Timer("simulation"):
        traj = sim.run(
            n_steps=args.n_steps,
            performance=args.performance,
            chunk_size=args.chunk_size,
        )

    print(f"Final positions shape : {sim.r.shape}")
    print(f"Trajectory shape      : {traj['positions'].shape}")

    if args.save:
        sim.save_trajectory(args.save)


if __name__ == "__main__":
    main()
