"""
ParaView source for LS-DYNA d3plot files, plain or FEMZIP-compressed.

ParaView 5.11's LS-DYNA reader cannot open FEMZIP files (SIDACT keeps the
d3plot name and header but compresses the results) and misreads the state
layout of these runs: it reports stresses the files do not contain and junk
plastic strains in some parts. This source reads both kinds with
lasso-python, which decompresses FEMZIP with the SIDACT library bundled in
lasso-python 1.5.x, and builds the deformed mesh of the requested time step.

Use it from paraview.simple; the source behaves like a reader with time steps:

    from d3plot_reader import open_d3plot
    reader = open_d3plot("/data/run/d3plot")

Output (vtkUnstructuredGrid, eroded elements removed):
    points        deformed coordinates of the time step
    point arrays  Deflection (displacement from the initial state), UserID (node ids)
    cell arrays   EffPlastStrn (shells; NaN for other elements), PartID
"""

import os
import sys
import types

import numpy as np
from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import (
    VTK_HEXAHEDRON,
    VTK_LINE,
    VTK_QUAD,
    vtkCellArray,
    vtkUnstructuredGrid,
)

# Decompressed arrays per file. A loaded run takes several hundred MB.
_ARRAYS = {}


def is_femzip_file(path):
    """True when lasso-python's FEMZIP API is available and recognizes the file."""
    try:
        from lasso.femzip import FemzipAPI
    except ImportError:
        return False
    try:
        return bool(FemzipAPI().is_sidact_file(str(path)))
    except Exception:
        return False


def _import_lasso_dyna():
    """
    Import lasso.dyna.

    Its binout module imports h5py, which fails inside a pvserver linked
    against another HDF5 than the h5py wheel. d3plots do not need h5py, so an
    empty module stands in for it there.
    """
    try:
        import h5py  # noqa: F401
    except ImportError:
        pass
    except Exception:
        for name in [name for name in sys.modules if name == "h5py" or name.startswith("h5py.")]:
            del sys.modules[name]
        sys.modules["h5py"] = types.ModuleType("h5py")
    import lasso.dyna

    return lasso.dyna


def is_d3plot(path):
    """True for d3plot files when lasso-python is installed to read them."""
    try:
        _import_lasso_dyna()
    except ImportError:
        return False
    return os.path.basename(str(path)) == "d3plot" or is_femzip_file(path)


def _arrays(path):
    """Result arrays of a d3plot (decompressed if needed), read once per file."""
    if path not in _ARRAYS:
        D3plot = _import_lasso_dyna().D3plot
        _ARRAYS[path] = D3plot(path, use_femzip=is_femzip_file(path)).arrays
    return _ARRAYS[path]


def release(path):
    """Drop the cached arrays of one file."""
    _ARRAYS.pop(str(path), None)


def release_all():
    """Drop the cached arrays of all files."""
    _ARRAYS.clear()


def request_information(algorithm, path):
    """Announce the time steps of the file (ProgrammableSource information script)."""
    executive = algorithm.GetExecutive()
    out_info = executive.GetOutputInformation(0)
    out_info.Remove(executive.TIME_STEPS())
    out_info.Remove(executive.TIME_RANGE())
    timesteps = [float(t) for t in _arrays(path)["timesteps"]]
    for t in timesteps:
        out_info.Append(executive.TIME_STEPS(), t)
    if timesteps:
        out_info.Append(executive.TIME_RANGE(), timesteps[0])
        out_info.Append(executive.TIME_RANGE(), timesteps[-1])


def part_table(path):
    """Parts of a d3plot as (id, name, element count), ordered by id."""
    arrays = _arrays(path)
    part_ids = arrays["part_ids"]
    title_ids = [int(i) for i in arrays.get("part_titles_ids", [])]
    # Title ids are either part ids or 1-based positions in part_ids.
    if title_ids and not set(title_ids) <= set(int(i) for i in part_ids):
        title_ids = [int(part_ids[i - 1]) for i in title_ids if 0 < i <= len(part_ids)]
    names = {
        part_id: title.decode(errors="replace").strip() if isinstance(title, bytes) else str(title).strip()
        for part_id, title in zip(title_ids, arrays.get("part_titles", []))
    }
    counts = {}
    for kind in ("shell", "solid", "beam", "tshell"):
        indexes = arrays.get(f"element_{kind}_part_indexes")
        if indexes is not None and len(indexes):
            for part_id, count in zip(*np.unique(part_ids[indexes], return_counts=True)):
                counts[int(part_id)] = counts.get(int(part_id), 0) + int(count)
    return [(part_id, names.get(part_id, ""), counts.get(part_id, 0)) for part_id in sorted(int(i) for i in part_ids)]


def request_data(algorithm, path, parts=None):
    """Produce the mesh of the requested time step (ProgrammableSource script)."""
    arrays = _arrays(path)
    executive = algorithm.GetExecutive()
    out_info = executive.GetOutputInformation(0)
    timesteps = arrays["timesteps"]
    step = 0
    if out_info.Has(executive.UPDATE_TIME_STEP()):
        step = int(np.abs(timesteps - out_info.Get(executive.UPDATE_TIME_STEP())).argmin())
    output = algorithm.GetOutput()
    output.ShallowCopy(_build_grid(arrays, step, parts))
    output.GetInformation().Set(output.DATA_TIME_STEP(), float(timesteps[step]))


def _data_script(path, parts):
    return f"import d3plot_reader; d3plot_reader.request_data(self, {path!r}, {parts!r})"


def open_d3plot(path, registration_name=None, parts=None):
    """Create a pipeline source that reads a d3plot, optionally only some part ids."""
    from paraview.simple import ProgrammableSource

    path = str(path)
    _arrays(path)  # decompress now so errors surface when loading
    return ProgrammableSource(
        registrationName=registration_name,
        OutputDataSetType="vtkUnstructuredGrid",
        ScriptRequestInformation=f"import d3plot_reader; d3plot_reader.request_information(self, {path!r})",
        Script=_data_script(path, sorted(parts) if parts is not None else None),
    )


def set_parts(source, path, parts):
    """Limit a source of open_d3plot to part ids; None shows all parts."""
    source.Script = _data_script(str(path), sorted(parts) if parts is not None else None)


def _build_grid(arrays, step, parts=None):
    """Mesh of one time step with the elements that are still alive (and in parts)."""
    part_ids = arrays["part_ids"]
    wanted = None if parts is None else np.asarray(list(parts))
    # (kind, VTK cell type, node columns used); beams keep their two end nodes.
    kinds = [("shell", VTK_QUAD, 4), ("solid", VTK_HEXAHEDRON, 8), ("beam", VTK_LINE, 2)]
    connectivity, cell_types, strain, parts = [], [], [], []
    used = np.zeros(len(arrays["node_coordinates"]), dtype=bool)
    for kind, cell_type, n_nodes in kinds:
        nodes = arrays.get(f"element_{kind}_node_indexes")
        if nodes is None or len(nodes) == 0:
            continue
        alive_key = f"element_{kind}_is_alive"
        alive = (
            arrays[alive_key][step] != 0
            if alive_key in arrays
            else np.ones(len(nodes), dtype=bool)
        )
        if wanted is not None:
            alive = alive & np.isin(part_ids[arrays[f"element_{kind}_part_indexes"]], wanted)
        nodes = nodes[alive, :n_nodes]
        used[nodes.ravel()] = True
        connectivity.append(np.hstack([np.full((len(nodes), 1), n_nodes), nodes]).ravel())
        cell_types.append(np.full(len(nodes), cell_type, dtype=np.uint8))
        if kind == "shell" and "element_shell_effective_plastic_strain" in arrays:
            # The mean over integration points, as LS-PrePost shows by default.
            strain.append(arrays["element_shell_effective_plastic_strain"][step][alive].mean(axis=1))
        else:
            strain.append(np.full(len(nodes), np.nan, dtype=np.float32))
        parts.append(part_ids[arrays[f"element_{kind}_part_indexes"][alive]])

    grid = vtkUnstructuredGrid()
    initial = arrays["node_coordinates"]
    # lasso names the deformed coordinates "node_displacement".
    current = arrays["node_displacement"][step] if "node_displacement" in arrays else initial
    # Nodes of eroded elements may fly off; they are not drawn, so keep them
    # in place to keep the bounds meaningful.
    current = np.where(used[:, None], current, initial)
    points = vtkPoints()
    points.SetData(numpy_to_vtk(np.ascontiguousarray(current, dtype=np.float32), deep=True))
    grid.SetPoints(points)
    cells = vtkCellArray()
    n_cells = sum(len(types) for types in cell_types)
    cells.SetCells(n_cells, numpy_to_vtkIdTypeArray(np.concatenate(connectivity).astype(np.int64), deep=True))
    grid.SetCells(numpy_to_vtk(np.concatenate(cell_types), deep=True), cells)

    # NaN keeps nodes of eroded elements out of color ranges.
    deflection_values = np.ascontiguousarray(current - initial, dtype=np.float32)
    deflection_values[~used] = np.nan
    deflection = numpy_to_vtk(deflection_values, deep=True)
    deflection.SetName("Deflection")
    grid.GetPointData().AddArray(deflection)
    strain_array = numpy_to_vtk(np.concatenate(strain).astype(np.float32), deep=True)
    strain_array.SetName("EffPlastStrn")
    grid.GetCellData().AddArray(strain_array)
    if "node_ids" in arrays:
        # Same name as ParaView's LS-DYNA reader uses for node ids.
        node_ids = numpy_to_vtk(np.asarray(arrays["node_ids"], dtype=np.int32), deep=True)
        node_ids.SetName("UserID")
        grid.GetPointData().AddArray(node_ids)
    part_array = numpy_to_vtk(np.concatenate(parts).astype(np.int32), deep=True)
    part_array.SetName("PartID")
    grid.GetCellData().AddArray(part_array)
    return grid
