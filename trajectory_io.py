from pathlib import Path

import numpy as np


def _build_argon_mdtraj_topology(num_atoms):
    import mdtraj as md

    topology = md.Topology()
    chain = topology.add_chain()
    residue = topology.add_residue("Ar", chain)
    for _ in range(num_atoms):
        topology.add_atom("Ar", md.element.argon, residue)
    return topology


def make_mdtraj_trajectory(positions_traj, box, length_scale_to_nm=0.1):
    """Convert a trajectory dict/array pair into an mdtraj.Trajectory."""
    import mdtraj as md

    xyz = np.asarray(positions_traj, dtype=np.float32)
    if xyz.ndim != 3 or xyz.shape[-1] != 3:
        raise ValueError("positions_traj must have shape (n_frames, n_atoms, 3).")

    box = np.asarray(box, dtype=np.float32)
    if box.shape != (3,):
        raise ValueError("box must have shape (3,) for mdtraj/nglview export.")

    n_frames, n_atoms, _ = xyz.shape
    topology = _build_argon_mdtraj_topology(n_atoms)

    return md.Trajectory(
        xyz=xyz * float(length_scale_to_nm),
        topology=topology,
        unitcell_lengths=np.tile(box * float(length_scale_to_nm), (n_frames, 1)),
        unitcell_angles=np.full((n_frames, 3), 90.0, dtype=np.float32),
    )


def view_trajectory_ngl(
    positions_traj,
    box,
    length_scale_to_nm=0.1,
    radius=0.3,
    representation="spacefill",
):
    """Create an nglview widget from a trajectory array."""
    import nglview as nv

    traj_md = make_mdtraj_trajectory(
        positions_traj,
        box,
        length_scale_to_nm=length_scale_to_nm,
    )

    view = nv.show_mdtraj(traj_md)
    if representation == "spacefill":
        view.clear_representations()
        view.add_spacefill(radius=radius)
    else:
        view.clear_representations()
        view.add_representation(representation)
    return view


def write_ovito_dump(positions_traj, box, filename="trajectory.dump"):
    """Write a LAMMPS-style dump file that OVITO can open directly."""
    positions_traj = np.asarray(positions_traj, dtype=float)
    box = np.asarray(box, dtype=float)

    if positions_traj.ndim != 3 or positions_traj.shape[-1] != 3:
        raise ValueError("positions_traj must have shape (n_frames, n_atoms, 3).")
    if box.shape != (3,):
        raise ValueError("box must have shape (3,) for dump export.")

    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for step, pos in enumerate(positions_traj):
            handle.write("ITEM: TIMESTEP\n")
            handle.write(f"{step}\n")
            handle.write("ITEM: NUMBER OF ATOMS\n")
            handle.write(f"{len(pos)}\n")
            handle.write("ITEM: BOX BOUNDS pp pp pp\n")
            for d in range(3):
                handle.write(f"0.0 {float(box[d]):.6f}\n")
            handle.write("ITEM: ATOMS id x y z\n")
            for atom_id, r in enumerate(pos, start=1):
                handle.write(f"{atom_id} {r[0]:.6f} {r[1]:.6f} {r[2]:.6f}\n")

    return output_path
