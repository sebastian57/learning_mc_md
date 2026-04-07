import argparse
import json
from pathlib import Path

import numpy as np
import pyvista as pv

from md import MD


def _parse_box(value):
    if value is None:
        return None
    try:
        box = np.array(json.loads(value), dtype=float)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Cannot parse box '{value}': use a JSON array.") from exc
    if box.shape not in {(3,), (3, 3)}:
        raise ValueError(f"Expected box shape (3,) or (3, 3), got {box.shape}.")
    return box


def _box_to_matrix(box):
    if box is None:
        return None
    box = np.asarray(box, dtype=float)
    if box.shape == (3,):
        return np.diag(box)
    if box.shape == (3, 3):
        return box
    raise ValueError(f"Expected box shape (3,) or (3, 3), got {box.shape}.")


def wrap_positions(positions, box):
    """Wrap positions into the periodic cell."""
    cell = _box_to_matrix(box)
    if cell is None:
        return np.asarray(positions, dtype=float)

    inv_cell = np.linalg.inv(cell)
    frac = np.asarray(positions, dtype=float) @ inv_cell.T
    frac = frac - np.floor(frac)
    return frac @ cell.T


def _cell_corners(box):
    cell = _box_to_matrix(box)
    origin = np.zeros(3, dtype=float)
    a, b, c = cell
    corners = np.array(
        [
            origin,
            a,
            b,
            c,
            a + b,
            a + c,
            b + c,
            a + b + c,
        ],
        dtype=float,
    )
    edges = np.array(
        [
            [0, 1], [0, 2], [0, 3],
            [1, 4], [1, 5],
            [2, 4], [2, 6],
            [3, 5], [3, 6],
            [4, 7], [5, 7], [6, 7],
        ],
        dtype=np.int32,
    )
    return corners, edges


def _make_box_mesh(box):
    corners, edges = _cell_corners(box)
    lines = []
    for start, end in edges:
        lines.extend([2, int(start), int(end)])
    return pv.PolyData(corners, lines=np.array(lines, dtype=np.int32))


def _write_pvd(collection_path, frame_paths, time_values):
    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
        "  <Collection>",
    ]
    for timestep, frame_path in zip(time_values, frame_paths):
        lines.append(
            f'    <DataSet timestep="{float(timestep):.6f}" group="" part="0" file="{frame_path}"/>'
        )
    lines.extend(["  </Collection>", "</VTKFile>"])
    collection_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_paraview_trajectory(traj, output_dir, box=None, every_nth=1, wrap=True):
    """Export sampled MD frames to ParaView-friendly VTK files."""
    if every_nth <= 0:
        raise ValueError("every_nth must be a positive integer.")

    output_dir = Path(output_dir)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    positions = np.asarray(traj["positions"])
    velocities = np.asarray(traj["velocities"]) if "velocities" in traj else None
    forces = np.asarray(traj["forces"]) if "forces" in traj else None
    temperatures = np.asarray(traj["temperature"]) if "temperature" in traj else None
    potential_energy = np.asarray(traj["potential_energy"]) if "potential_energy" in traj else None

    sampled_steps = list(range(0, positions.shape[0], every_nth))
    written_frames = []

    for frame_id, step_idx in enumerate(sampled_steps):
        xyz = positions[step_idx]
        if wrap and box is not None:
            xyz = wrap_positions(xyz, box)

        cloud = pv.PolyData(xyz)
        n_particles = xyz.shape[0]
        cloud["particle_id"] = np.arange(n_particles, dtype=np.int32)
        cloud["step"] = np.full(n_particles, step_idx, dtype=np.int32)

        if velocities is not None:
            cloud["velocity"] = velocities[step_idx]
            cloud["speed"] = np.linalg.norm(velocities[step_idx], axis=-1)
        if forces is not None:
            cloud["force"] = forces[step_idx]
            cloud["force_norm"] = np.linalg.norm(forces[step_idx], axis=-1)
        if temperatures is not None:
            cloud["temperature"] = np.full(n_particles, temperatures[step_idx], dtype=float)
        if potential_energy is not None:
            cloud["potential_energy"] = np.full(n_particles, potential_energy[step_idx], dtype=float)

        frame_path = frames_dir / f"frame_{frame_id:05d}.vtp"
        cloud.save(frame_path)
        written_frames.append(frame_path.relative_to(output_dir).as_posix())

    if box is not None:
        box_mesh = _make_box_mesh(box)
        box_mesh.save(output_dir / "box.vtp")

    _write_pvd(output_dir / "trajectory.pvd", written_frames, sampled_steps)

    metadata = {
        "n_input_frames": int(positions.shape[0]),
        "n_exported_frames": int(len(sampled_steps)),
        "every_nth": int(every_nth),
        "box": None if box is None else np.asarray(box, dtype=float).tolist(),
        "wrapped": bool(wrap),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    return {
        "output_dir": output_dir,
        "frames_dir": frames_dir,
        "pvd_path": output_dir / "trajectory.pvd",
        "metadata_path": output_dir / "metadata.json",
        "n_exported_frames": len(sampled_steps),
    }


def main():
    parser = argparse.ArgumentParser(description="Export MD trajectories to ParaView/VTK.")
    parser.add_argument("--trajectory", required=True, help="Path to a saved .npz trajectory.")
    parser.add_argument("--output-dir", default="paraview_trajectory",
                        help="Directory where VTK files will be written.")
    parser.add_argument("--box", default=None,
                        help="Periodic box as JSON array with shape (3,) or (3, 3).")
    parser.add_argument("--every-nth", type=int, default=1,
                        help="Export every nth frame.")
    parser.add_argument("--no-wrap", action="store_true",
                        help="Disable wrapping positions into the periodic box before export.")
    args = parser.parse_args()

    traj = MD.load_trajectory(args.trajectory)
    result = export_paraview_trajectory(
        traj,
        output_dir=args.output_dir,
        box=_parse_box(args.box),
        every_nth=args.every_nth,
        wrap=not args.no_wrap,
    )

    print(f"Exported {result['n_exported_frames']} frames to {result['output_dir']}")
    print(f"Open {result['pvd_path']} in ParaView.")


if __name__ == "__main__":
    main()
