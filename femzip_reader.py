"""
ParaView reader for FEMZIP-compressed LS-DYNA d3plot files.

FEMZIP (SIDACT) keeps the d3plot name and header but compresses the results,
which ParaView's LS-DYNA reader cannot read. This reader decompresses them with
lasso-python's FEMZIP API (lasso-python 1.5.x bundles the SIDACT library) and
builds the deformed mesh of the requested time step.

Use it from paraview.simple; the source behaves like a reader with time steps:

    from femzip_reader import open_femzip_d3plot
    reader = open_femzip_d3plot("/data/run/d3plot")

Output (vtkUnstructuredGrid, eroded elements removed):
    points        deformed coordinates of the time step
    point arrays  Deflection (displacement from the initial state)
    cell arrays   EffPlastStrn (shells; NaN for other elements), PartID
"""

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


def _arrays(path):
    """Decompressed result arrays of a FEMZIP d3plot, read once per file."""
    if path not in _ARRAYS:
        from lasso.dyna import D3plot

        _ARRAYS[path] = D3plot(path, use_femzip=True).arrays
    return _ARRAYS[path]


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


def request_data(algorithm, path):
    """Produce the mesh of the requested time step (ProgrammableSource script)."""
    arrays = _arrays(path)
    executive = algorithm.GetExecutive()
    out_info = executive.GetOutputInformation(0)
    timesteps = arrays["timesteps"]
    step = 0
    if out_info.Has(executive.UPDATE_TIME_STEP()):
        step = int(np.abs(timesteps - out_info.Get(executive.UPDATE_TIME_STEP())).argmin())
    output = algorithm.GetOutput()
    output.ShallowCopy(_build_grid(arrays, step))
    output.GetInformation().Set(output.DATA_TIME_STEP(), float(timesteps[step]))


def open_femzip_d3plot(path, registration_name=None):
    """Create a pipeline source that reads a FEMZIP d3plot."""
    from paraview.simple import ProgrammableSource

    path = str(path)
    _arrays(path)  # decompress now so errors surface when loading
    call = f"import femzip_reader; femzip_reader.{{}}(self, {path!r})"
    return ProgrammableSource(
        registrationName=registration_name,
        OutputDataSetType="vtkUnstructuredGrid",
        ScriptRequestInformation=call.format("request_information"),
        Script=call.format("request_data"),
    )


def _build_grid(arrays, step):
    """Mesh of one time step with the elements that are still alive."""
    part_ids = arrays["part_ids"]
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
    points = vtkPoints()
    points.SetData(numpy_to_vtk(np.ascontiguousarray(current, dtype=np.float32), deep=True))
    grid.SetPoints(points)
    cells = vtkCellArray()
    n_cells = sum(len(types) for types in cell_types)
    cells.SetCells(n_cells, numpy_to_vtkIdTypeArray(np.concatenate(connectivity).astype(np.int64), deep=True))
    grid.SetCells(numpy_to_vtk(np.concatenate(cell_types), deep=True), cells)

    # Nodes of eroded elements may fly off; NaN keeps them out of color ranges.
    deflection_values = np.ascontiguousarray(current - initial, dtype=np.float32)
    deflection_values[~used] = np.nan
    deflection = numpy_to_vtk(deflection_values, deep=True)
    deflection.SetName("Deflection")
    grid.GetPointData().AddArray(deflection)
    strain_array = numpy_to_vtk(np.concatenate(strain).astype(np.float32), deep=True)
    strain_array.SetName("EffPlastStrn")
    grid.GetCellData().AddArray(strain_array)
    part_array = numpy_to_vtk(np.concatenate(parts).astype(np.int32), deep=True)
    part_array.SetName("PartID")
    grid.GetCellData().AddArray(part_array)
    return grid
