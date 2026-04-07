import argparse
import importlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import jax
import jax.numpy as jnp

from neighbor_list import build_neighbor_list, minimum_image, validate_neighbor_list_mode
from potentials import HarmonicPotential


# ---------------------------------------------------------------------------
# Timer
# ---------------------------------------------------------------------------


def _get_tqdm(backend="auto"):
    """Return a tqdm implementation suited to the active frontend."""
    if backend == "none":
        return None

    if backend == "terminal":
        from tqdm import tqdm as terminal_tqdm
        return terminal_tqdm

    if backend == "notebook":
        from tqdm.notebook import tqdm as notebook_tqdm
        return notebook_tqdm

    if backend != "auto":
        raise ValueError(
            "progress_backend must be one of 'auto', 'notebook', 'terminal', or 'none'."
        )

    try:
        from IPython import get_ipython

        shell = get_ipython()
        if shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell":
            from tqdm.notebook import tqdm as notebook_tqdm
            return notebook_tqdm
    except Exception:
        pass

    from tqdm.auto import tqdm as auto_tqdm
    return auto_tqdm


def _make_progress_bar(total, enabled=True, backend="auto"):
    """Create a manually-updated progress bar for chunked execution."""
    if not enabled or total <= 0:
        return None

    tqdm_factory = _get_tqdm(backend=backend)
    if tqdm_factory is None:
        return None

    return tqdm_factory(
        total=total,
        desc="MD simulation",
        unit="chunk",
        dynamic_ncols=True,
        leave=True,
        mininterval=0.1,
        bar_format="{l_bar}{bar}| chunk {n_fmt}/{total_fmt}  [{elapsed}<{remaining}, {rate_fmt}]",
    )

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
# MD engine
# ---------------------------------------------------------------------------

class MD:
    """Velocity-Verlet molecular dynamics engine.

    Parameters
    ----------
    r         : array (N, 3)   initial positions
    v         : array (N, 3)   initial velocities
    a         : array (N, 3)   initial accelerations
    potential : callable  potential callable; with a neighbor list it should accept
                          (positions, neighbor_list, box)
    neighbor_list_mode : str  one of "none", "single", "many-body"
    neighbor_list_pbc : bool  whether to apply periodic wrapping in neighbor list geometry
    neighbor_list_padded : bool  whether to build padded per-particle neighbor indices
    neighbor_list_max_neighbors : optional fixed padding width
    dt        : float     timestep
    m         : float     particle mass (uniform)

    After run(), these attributes are available:
        self.r            final positions
        self.v            final velocities
        self.a            final accelerations
        self.trajectory   dict with keys 'positions', 'velocities', 'accelerations',
                          'forces', and 'potential_energy'
    """

    def __init__(
        self,
        r,
        v,
        a,
        potential,
        dt=0.01,
        m=1.0,
        neighbor_list_mode="none",
        box=None,
        r_cut=None,
        r_skin=0.0,
        neighbor_list_pbc=False,
        neighbor_list_padded=True,
        neighbor_list_max_neighbors=None,
    ):
        self.r = jnp.array(r, dtype=jnp.float32)
        self.v = jnp.array(v, dtype=jnp.float32)
        self.a = jnp.array(a, dtype=jnp.float32)
        self.potential = potential
        self.dt = float(dt)
        self.m = float(m)
        self.neighbor_list_mode = validate_neighbor_list_mode(neighbor_list_mode)
        self.box = None if box is None else jnp.array(box, dtype=jnp.float32)
        self.r_cut = r_cut
        self.r_skin = float(r_skin)
        self.neighbor_list_pbc = bool(neighbor_list_pbc)
        self.neighbor_list_padded = bool(neighbor_list_padded)
        self.neighbor_list_max_neighbors = neighbor_list_max_neighbors
        if (
            self.neighbor_list_mode != "none"
            and self.neighbor_list_padded
            and self.neighbor_list_max_neighbors is None
        ):
            raise ValueError(
                "Padded neighbor lists inside MD require neighbor_list_max_neighbors "
                "to be set to a fixed integer width."
            )
        self.neighbor_list = None
        self.reference_positions = None
        self.trajectory = None

    @staticmethod
    def initialize_velocities(
        n_particles,
        temperature,
        m=1.0,
        dim=3,
        rng_key=None,
        remove_drift=True,
        dtype=jnp.float32,
        k_B=1.0,
    ):
        """Sample initial velocities from a Maxwell-Boltzmann distribution."""
        if rng_key is None:
            rng_key = jax.random.PRNGKey(0)

        std = jnp.sqrt(k_B * float(temperature) / float(m))
        velocities = std * jax.random.normal(rng_key, (n_particles, dim), dtype=dtype)

        if remove_drift:
            velocities = velocities - jnp.mean(velocities, axis=0, keepdims=True)

        return velocities

    def _kinetic_energy(self, velocities):
        return 0.5 * self.m * jnp.sum(velocities ** 2)

    def _temperature(self, velocities):
        n_particles = velocities.shape[0]
        dof = max(1, 3 * n_particles - 3)
        return 2.0 * self._kinetic_energy(velocities) / dof

    def _wrap_positions(self, positions):
        if not self.neighbor_list_pbc or self.box is None:
            return positions

        box = jnp.asarray(self.box, dtype=positions.dtype)
        if box.ndim == 1:
            return positions - box * jnp.floor(positions / box)
        if box.shape == (3, 3):
            inv_box = jnp.linalg.inv(box)
            fractional = jnp.einsum("...i,ij->...j", positions, inv_box)
            fractional = fractional - jnp.floor(fractional)
            return jnp.einsum("...i,ij->...j", fractional, box)

        raise ValueError(f"Expected box shape (3,) or (3, 3), got {box.shape}.")

    # ------------------------------------------------------------------
    # Internal step (built once, reused across chunks)
    # ------------------------------------------------------------------

    def _build_neighbor_list(self, positions):
        return build_neighbor_list(
            positions,
            box=self.box,
            cutoff=self.r_cut,
            skin=self.r_skin,
            mode=self.neighbor_list_mode,
            pbc=self.neighbor_list_pbc,
            padded=self.neighbor_list_padded,
            max_neighbors=self.neighbor_list_max_neighbors,
        )

    def _get_neighbor_list(self, positions):
        return self._build_neighbor_list(positions)

    def _should_rebuild_neighbor_list(self, positions):
        if self.neighbor_list_mode == "none":
            return False
        if self.neighbor_list is None or self.reference_positions is None:
            return True
        if self.r_skin <= 0.0:
            return True

        displacement = positions - self.reference_positions
        if self.neighbor_list_pbc:
            displacement = minimum_image(displacement, box=self.box)

        max_displacement = jnp.max(jnp.linalg.norm(displacement, axis=-1))
        return bool(max_displacement >= 0.5 * self.r_skin)

    def _update_neighbor_list(self, positions, force=False):
        if self.neighbor_list_mode == "none":
            self.neighbor_list = None
            self.reference_positions = None
            return None

        if force or self._should_rebuild_neighbor_list(positions):
            self.neighbor_list = self._build_neighbor_list(positions)
            self.reference_positions = jnp.array(positions)

        return self.neighbor_list

    def _build_single_particle_force_fn(self):
        return jax.vmap(lambda r_i: -jax.grad(self.potential)(r_i))

    def _build_single_particle_energy_fn(self):
        return lambda positions: jnp.sum(jax.vmap(self.potential)(positions))

    def _build_neighbor_list_force_fn(self, neighbor_list):
        energy_fn = jax.value_and_grad(
            lambda positions, neighbor_list: self.potential(positions, neighbor_list, self.box)
        )

        def force_fn(r):
            _, grad = energy_fn(r, neighbor_list)
            return -grad

        return force_fn

    def _build_neighbor_list_energy_fn(self, neighbor_list):
        return lambda positions: self.potential(positions, neighbor_list, self.box)

    def _build_force_fn(self, neighbor_list=None):
        if self.neighbor_list_mode == "none":
            return self._build_single_particle_force_fn()
        return self._build_neighbor_list_force_fn(neighbor_list)

    def _build_energy_force_fn(self, neighbor_list=None):
        if self.neighbor_list_mode == "none":
            energy_fn = self._build_single_particle_energy_fn()
        else:
            energy_fn = self._build_neighbor_list_energy_fn(neighbor_list)

        energy_and_grad = jax.value_and_grad(energy_fn)

        def energy_force_fn(positions):
            energy, grad = energy_and_grad(positions)
            return energy, -grad

        return energy_force_fn

    def _compute_initial_acceleration(self, positions):
        wrapped_positions = self._wrap_positions(positions)
        if self.neighbor_list_mode == "none":
            neighbor_list = None
        else:
            neighbor_list = self._update_neighbor_list(wrapped_positions, force=True)
        energy_force_fn = self._build_energy_force_fn(neighbor_list)
        _, force = energy_force_fn(wrapped_positions)
        return wrapped_positions, force / self.m

    def _build_step_fn(self, neighbor_list=None):
        energy_force_fn = self._build_energy_force_fn(neighbor_list)
        dt, m = self.dt, self.m

        @jax.jit
        def step(carry, _):
            r, v, a = carry
            r_ = r + v * dt + 0.5 * a * dt ** 2
            r_ = self._wrap_positions(r_)
            energy_, force_ = energy_force_fn(r_)
            a_ = force_ / m
            v_ = v + 0.5 * (a + a_) * dt
            kinetic_energy_ = self._kinetic_energy(v_)
            temperature_ = self._temperature(v_)
            return (r_, v_, a_), (r_, v_, a_, force_, energy_, kinetic_energy_, temperature_)

        return step

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, n_steps=100, performance=False, chunk_size=10, progress_backend="auto"):
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
        progress_backend : one of "auto", "notebook", "terminal", or "none".
        """
        self.r, self.a = self._compute_initial_acceleration(self.r)

        if self.neighbor_list_mode == "none":
            step = self._build_step_fn()

            if performance:
                (r_f, v_f, a_f), (r_t, v_t, a_t, f_t, e_t, ke_t, temp_t) = jax.lax.scan(
                    step, (self.r, self.v, self.a), None, length=n_steps
                )
                self.r, self.v, self.a = r_f, v_f, a_f
                self.trajectory = {
                    "positions":     np.array(r_t),
                    "velocities":    np.array(v_t),
                    "accelerations": np.array(a_t),
                    "forces":        np.array(f_t),
                    "potential_energy": np.array(e_t),
                    "kinetic_energy": np.array(ke_t),
                    "temperature": np.array(temp_t),
                }
            else:
                n_full, remainder = divmod(n_steps, chunk_size)
                chunk_sizes = [chunk_size] * n_full
                if remainder:
                    chunk_sizes.append(remainder)

                carry = (self.r, self.v, self.a)
                all_r, all_v, all_a, all_f, all_e, all_ke, all_temp = [], [], [], [], [], [], []

                bar = _make_progress_bar(
                    total=len(chunk_sizes),
                    enabled=True,
                    backend=progress_backend,
                )

                try:
                    for idx, cs in enumerate(chunk_sizes, start=1):
                        carry, (r_chunk, v_chunk, a_chunk, f_chunk, e_chunk, ke_chunk, temp_chunk) = jax.lax.scan(
                            step, carry, None, length=cs
                        )
                        jax.block_until_ready(carry)
                        all_r.append(np.array(r_chunk))
                        all_v.append(np.array(v_chunk))
                        all_a.append(np.array(a_chunk))
                        all_f.append(np.array(f_chunk))
                        all_e.append(np.array(e_chunk))
                        all_ke.append(np.array(ke_chunk))
                        all_temp.append(np.array(temp_chunk))
                        if bar is not None:
                            bar.update(1)
                            bar.set_postfix_str(f"steps={min(idx * chunk_size, n_steps)}/{n_steps}")
                            bar.refresh()
                finally:
                    if bar is not None:
                        bar.close()

                self.r, self.v, self.a = carry
                self.trajectory = {
                    "positions":     np.concatenate(all_r, axis=0),
                    "velocities":    np.concatenate(all_v, axis=0),
                    "accelerations": np.concatenate(all_a, axis=0),
                    "forces":        np.concatenate(all_f, axis=0),
                    "potential_energy": np.concatenate(all_e, axis=0),
                    "kinetic_energy": np.concatenate(all_ke, axis=0),
                    "temperature": np.concatenate(all_temp, axis=0),
                }

            self.neighbor_list = None
            self.reference_positions = None
            return self.trajectory

        n_full, remainder = divmod(n_steps, chunk_size)
        chunk_sizes = [chunk_size] * n_full
        if remainder:
            chunk_sizes.append(remainder)

        carry = (self.r, self.v, self.a)
        all_r, all_v, all_a, all_f, all_e, all_ke, all_temp = [], [], [], [], [], [], []

        bar = _make_progress_bar(
            total=len(chunk_sizes),
            enabled=not performance,
            backend=progress_backend,
        )

        self._update_neighbor_list(self.r, force=True)

        try:
            for idx, cs in enumerate(chunk_sizes, start=1):
                current_r, _, _ = carry
                neighbor_list = self._update_neighbor_list(current_r)
                step = self._build_step_fn(neighbor_list)
                carry, (r_chunk, v_chunk, a_chunk, f_chunk, e_chunk, ke_chunk, temp_chunk) = jax.lax.scan(
                    step, carry, None, length=cs
                )
                jax.block_until_ready(carry)
                all_r.append(np.array(r_chunk))
                all_v.append(np.array(v_chunk))
                all_a.append(np.array(a_chunk))
                all_f.append(np.array(f_chunk))
                all_e.append(np.array(e_chunk))
                all_ke.append(np.array(ke_chunk))
                all_temp.append(np.array(temp_chunk))
                if bar is not None:
                    bar.update(1)
                    bar.set_postfix_str(f"steps={min(idx * chunk_size, n_steps)}/{n_steps}")
                    bar.refresh()
        finally:
            if bar is not None:
                bar.close()

        self.r, self.v, self.a = carry
        self.trajectory = {
            "positions":     np.concatenate(all_r, axis=0),
            "velocities":    np.concatenate(all_v, axis=0),
            "accelerations": np.concatenate(all_a, axis=0),
            "forces":        np.concatenate(all_f, axis=0),
            "potential_energy": np.concatenate(all_e, axis=0),
            "kinetic_energy": np.concatenate(all_ke, axis=0),
            "temperature": np.concatenate(all_temp, axis=0),
        }
        self.neighbor_list = self._build_neighbor_list(self.r)
        self.reference_positions = jnp.array(self.r)
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

        Returns a dict with the saved trajectory arrays.
        """
        data = np.load(path if path.endswith(".npz") else path + ".npz")
        return {k: data[k] for k in data.files}


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


def _parse_box(value):
    if value is None:
        return None
    try:
        arr = jnp.array(json.loads(value), dtype=jnp.float32)
    except json.JSONDecodeError:
        raise ValueError(f"Cannot parse box '{value}': use a JSON array.")
    if arr.shape not in {(3,), (3, 3)}:
        raise ValueError(f"Expected box shape (3,) or (3, 3), got {arr.shape}.")
    return arr


def _load_module_from_path(path):
    spec = importlib.util.spec_from_file_location("custom_potential", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_potential(source, name="potential"):
    if source is None:
        return HarmonicPotential()

    if Path(source).suffix == ".py" or Path(source).exists():
        mod = _load_module_from_path(source)
    else:
        mod = importlib.import_module(source)

    if not hasattr(mod, name):
        raise AttributeError(f"{source} must define a callable named '{name}'.")

    potential = getattr(mod, name)
    if not callable(potential):
        raise TypeError(f"{source}:{name} must be callable.")
    return potential


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
    parser.add_argument("--progress-backend", default="auto",
                        choices=("auto", "notebook", "terminal", "none"),
                        help="Progress display backend for chunked runs.")
    parser.add_argument("--potential",     default=None,
                        help="Python file or importable module containing a potential.")
    parser.add_argument("--potential-name", default="potential",
                        help="Callable name to load from --potential (default: potential).")
    parser.add_argument("--neighbor-list-mode", default="none",
                        choices=("none", "single", "many-body"),
                        help="Use no neighbor list for single-particle potentials, or build a single/many-body topology for full-configuration potentials.")
    parser.add_argument("--box",           default=None,
                        help="Periodic box as JSON array with shape (3,) or (3, 3).")
    parser.add_argument("--r-cut",         type=float, default=None,
                        help="Neighbor-list cutoff radius placeholder.")
    parser.add_argument("--r-skin",        type=float, default=0.0,
                        help="Neighbor-list skin radius placeholder.")
    parser.add_argument("--neighbor-list-pbc", action="store_true",
                        help="Apply periodic boundary conditions in neighbor-list geometry.")
    parser.add_argument("--neighbor-list-unpadded", action="store_true",
                        help="Build an unpadded neighbor list instead of padded per-particle indices.")
    parser.add_argument("--neighbor-list-max-neighbors", type=int, default=None,
                        help="Fixed padded width for neighbor-list construction.")
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
    box = _parse_box(args.box)

    potential = _load_potential(args.potential, name=args.potential_name)

    sim = MD(
        r0,
        v0,
        a0,
        potential,
        dt=args.dt,
        m=args.m,
        neighbor_list_mode=args.neighbor_list_mode,
        box=box,
        r_cut=args.r_cut,
        r_skin=args.r_skin,
        neighbor_list_pbc=args.neighbor_list_pbc,
        neighbor_list_padded=not args.neighbor_list_unpadded,
        neighbor_list_max_neighbors=args.neighbor_list_max_neighbors,
    )

    with Timer("simulation"):
        traj = sim.run(
            n_steps=args.n_steps,
            performance=args.performance,
            chunk_size=args.chunk_size,
            progress_backend=args.progress_backend,
        )

    print(f"Final positions shape : {sim.r.shape}")
    print(f"Trajectory shape      : {traj['positions'].shape}")

    if args.save:
        sim.save_trajectory(args.save)


if __name__ == "__main__":
    main()
