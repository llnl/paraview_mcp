"""
Time series of LS-DYNA runs from their binout files.

binout holds the solver's history output: global and part energies (glstat,
matsum), node results (nodout), section, contact and rigid wall forces
(secforc, rcforc, rwforc), sliding interface energies (sleout) and joint
forces (jntforc). It is read with lasso-python.

Series are returned as data the chat client draws as a chart
(SERIES_MEDIA_TYPE) plus a short text summary for the model.
"""

import glob
import json
import os

import numpy as np

SERIES_MEDIA_TYPE = "application/vnd.onion.series+json"

# Fixed-width entity names in binout legends.
_LEGEND_WIDTH = 80
# Arrays that describe a category instead of holding results over time.
_METADATA_KEYS = {
    "time", "ids", "legend", "legend_ids", "date", "version", "revision",
    "build_id", "title", "cycle", "setid", "side", "singlesided", "tie_count",
    "element_types", "ts_eltype", "ts_element", "ts_part",
}

_BINOUTS = {}


def run_directory(path):
    """The run directory of a run directory or of a file inside it."""
    return path if os.path.isdir(path) else os.path.dirname(path)


def _binout(run_dir):
    if run_dir not in _BINOUTS:
        from lasso.dyna import Binout

        pattern = os.path.join(run_dir, "binout*")
        if not glob.glob(pattern):
            raise FileNotFoundError(f"No binout files in {run_dir}")
        _BINOUTS[run_dir] = Binout(pattern)
    return _BINOUTS[run_dir]


def _read(binout, path, *keys):
    return binout.read(*path.split("/"), *keys)


def _categories(binout, path=None):
    """Category paths that contain a time array, e.g. 'nodout' or 'rwforc/forces'."""
    found = []
    for key in _read(binout, path) if path else binout.read():
        child = f"{path}/{key}" if path else key
        try:
            entries = _read(binout, child)
        except Exception:
            continue
        if not isinstance(entries, list):
            continue
        if "time" in entries:
            found.append(child)
        elif child.count("/") < 2:
            found.extend(_categories(binout, child))
    return found


def _entity_names(binout, path, ids):
    """Legend names per entity, falling back to their ids."""
    names = [str(entity_id) for entity_id in ids]
    try:
        legend = _read(binout, path, "legend")
    except Exception:
        return names
    if isinstance(legend, str):
        chunks = [legend[i:i + _LEGEND_WIDTH].strip() for i in range(0, len(legend), _LEGEND_WIDTH)]
        if len(chunks) >= len(ids):
            names = [chunk or name for chunk, name in zip(chunks, names)]
    return names


def _variables(binout, path, n_times):
    """Result arrays of a category with one row per output time."""
    variables = []
    for key in _read(binout, path):
        if key in _METADATA_KEYS:
            continue
        try:
            value = _read(binout, path, key)
        except Exception:
            continue
        if isinstance(value, np.ndarray) and value.dtype.kind in "fiu" and value.ndim in (1, 2) and len(value) == n_times:
            variables.append(key)
    return sorted(variables)


def _ids(binout, path):
    try:
        ids = _read(binout, path, "ids")
    except Exception:
        try:
            ids = _read(binout, path, "legend_ids")
        except Exception:
            return []
    return [int(i) for i in np.atleast_1d(ids)]


def list_time_series(run_path):
    """Describe the time series of a run for the model."""
    binout = _binout(run_directory(run_path))
    lines = []
    for path in _categories(binout):
        time = np.asarray(_read(binout, path, "time"))
        variables = _variables(binout, path, len(time))
        if not variables:
            continue
        ids = _ids(binout, path)
        names = _entity_names(binout, path, ids)
        entities = ""
        if len(ids) > 1:
            shown = ", ".join(names[:15]) + (", ..." if len(names) > 15 else "")
            entities = f"\n  {len(ids)} entities: {shown}"
        lines.append(
            f"{path} ({len(time)} samples, t={time[0]:g}..{time[-1]:g}): "
            f"{', '.join(variables)}{entities}"
        )
    return "\n".join(lines)


def _select_entities(names, ids, wanted):
    """Column indexes of entities matching an id or name, else a part of a name (case-insensitive)."""
    if not wanted:
        return list(range(len(ids)))
    selected = []
    for item in wanted:
        text = str(item).strip().lower()
        matches = [
            index for index, (name, entity_id) in enumerate(zip(names, ids))
            if text in (str(entity_id), name.lower())
        ] or [index for index, name in enumerate(names) if text in name.lower()]
        if not matches:
            raise ValueError(f"No entity matches {item!r}; known: {', '.join(names[:30])}")
        selected.extend(index for index in matches if index not in selected)
    return selected


def _summary(name, time, values):
    finite = np.isfinite(values)
    if not finite.any():
        return f"- {name}: no values"
    peak = int(np.nanargmax(np.abs(np.where(finite, values, 0))))
    return (
        f"- {name}: min {np.nanmin(values):.4g}, max {np.nanmax(values):.4g}, "
        f"largest magnitude {values[peak]:.4g} at t={time[peak]:.4g}, final {values[finite][-1]:.4g}"
    )


def get_time_series(run_paths, category, variables, entities=None, title=None, max_points=2000):
    """
    Collect series of one category over runs, variables and entities.

    The first run's output times are the x axis; other runs are interpolated
    onto it. Longer series are thinned to max_points.

    Returns:
        tuple: (summary text, payload dict for SERIES_MEDIA_TYPE)
    """
    series = []
    x = None
    for run_path in run_paths:
        run_dir = run_directory(run_path)
        binout = _binout(run_dir)
        time = np.asarray(_read(binout, category, "time"), dtype=float)
        available = _variables(binout, category, len(time))
        ids = _ids(binout, category)
        names = _entity_names(binout, category, ids)
        if x is None:
            x = time
        for variable in variables:
            if variable not in available:
                raise ValueError(f"{category} has no {variable!r}; available: {', '.join(available)}")
            values = np.asarray(_read(binout, category, variable), dtype=float)
            columns = [values] if values.ndim == 1 else [values[:, index] for index in _select_entities(names, ids, entities)]
            labels = [None] if values.ndim == 1 else [names[index] for index in _select_entities(names, ids, entities)]
            for column, entity in zip(columns, labels):
                if time is not x:
                    column = np.interp(x, time, column, left=np.nan, right=np.nan)
                parts = [os.path.basename(run_dir)] if len(run_paths) > 1 else []
                parts += [entity] if entity else []
                parts += [variable] if len(variables) > 1 or not parts else []
                series.append({"name": " ".join(parts), "values": column})

    if len(x) > max_points:
        keep = np.unique(np.linspace(0, len(x) - 1, max_points).round().astype(int))
        x = x[keep]
        for item in series:
            item["values"] = item["values"][keep]

    title = title or f"{category} {', '.join(variables)}"
    summary = "\n".join(
        [f"{title}: {len(series)} series, {len(x)} points, t={x[0]:g}..{x[-1]:g}"]
        + [_summary(item["name"], x, item["values"]) for item in series[:20]]
    )
    payload = {
        "title": title,
        "x": {"label": "Time", "values": [float(v) for v in x]},
        "series": [
            {"name": item["name"], "values": [None if not np.isfinite(v) else float(v) for v in item["values"]]}
            for item in series
        ],
    }
    return summary, payload


def payload_bytes(payload):
    return json.dumps(payload, separators=(",", ":")).encode()
