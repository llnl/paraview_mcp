"""
ParaView MCP Server

This script runs as a standalone process and:
1. Connects to ParaView using its Python API over network
2. Exposes key ParaView functionality through the MCP protocol
3. Updates visualizations in the existing ParaView viewport

Usage:
1. Start pvserver with --multi-clients flag (e.g., pvserver --multi-clients --server-port=11111)
2. Start ParaView app and connect to the server
3. Configure Claude Desktop to use this script

Headless usage (no GUI, e.g. in a container):
    python paraview_mcp_server.py --server builtin --transport streamable-http \
        --http-host 0.0.0.0 --http-port 8765 --data-root /data --output-dir /output

"""
import base64
import os
import sys
import logging
import argparse
from pathlib import Path

from mcp.server.fastmcp import FastMCP, Image
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import BlobResourceContents, EmbeddedResource
from paraview_manager import ParaViewManager
import time_series

# Configure logging
log_dir = Path(os.environ.get("PARAVIEW_MCP_LOG_DIR", Path.home() / "paraview_logs"))
os.makedirs(log_dir, exist_ok=True)
log_file = log_dir / "paraview_mcp_external.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

# Default prompt that instructs Claude how to interact with ParaView
default_prompt = """
When using ParaView through this interface, please follow these guidelines:

1. IMPORTANT: Only call strictly necessary ParaView functions per reply (and please limit the total number of call per reply). This ensures operations execute in a more interative manner and no excessive calls to related but non-essential functions. 

2. The only execute multiple repeated function call when given a target goal (e.g., identify a specific object), where different parameters need to used (e.g., isosurface with different isovalue). Avoid repeated calling of color map function unless user specific ask for color map design.

3. Paraview will be connect to mcp server on starup so no need to connect first.

4. Use list_data_files to find data files. For transient results (e.g. LS-DYNA d3plot crash simulations), use get_timesteps and set_time to choose the time step to show.

5. Call get_screenshot to show the user the current view after changing it. For transient results, render_animation shows how the result evolves over time. compare_runs shows design variants side by side.

6. LS-DYNA d3plot results, plain or FEMZIP-compressed, provide the point arrays 'Deflection' (displacement) and 'UserID' (node ids) and the cell arrays 'EffPlastStrn' (effective plastic strain of shells; other elements have none) and 'PartID'. Eroded elements are removed.

7. For values at specific places, use get_data_info for bounds and ranges, sample_over_line for a profile along a line, and probe_over_time for a node over time. Time series (energies, forces, accelerations) come from each run's binout files: find them with list_time_series and chart them with get_time_series, also across runs to compare them. The runs appear to use the units mm, s, t (tonnes), N and mJ (N*mm), inferred from the initial velocity and kinetic energy.

"""
    
logger = logging.getLogger("pv_external_mcp")

# Create the ParaView manager
pv_manager = ParaViewManager()

# Initialize FastMCP server for Claude Desktop integration with default prompt
mcp = FastMCP("ParaView", instructions=default_prompt)

# ============================================================================
# MCP Tools for ParaView
# ============================================================================

@mcp.tool()
def load_data(file_path: str) -> str:
    """
    Load data from a file into ParaView.
    
    Args:
        file_path: Path to the data file (supports VTK, EXODUS, CSV, RAW, etc.)
    
    Returns:
        Status message
    """
    success, message, _, source_name = pv_manager.load_data(file_path)
    if success:
        return f"{message}. Source registered as '{source_name}'."
    else:
        return message

@mcp.tool()
def save_contour_as_stl(stl_filename: str = "contour.stl") -> str:
    """
    Save the currently active contour (or any surface/mesh source) as an STL file
    in the same folder as the originally loaded data.

    Args:
        stl_filename: The STL file name to use, defaults to 'contour.stl'.

    Returns:
        A status message (string).
    """
    success, message, path = pv_manager.save_contour_as_stl(stl_filename)
    return message

@mcp.tool()
def create_source(source_type: str) -> str:
    """
    Create a new geometric source.
    
    Args:
        source_type: Type of source to create (Sphere, Cone, Cylinder, Plane, Box)
    
    Returns:
        Status message
    """
    success, message, _, source_name = pv_manager.create_source(source_type)
    if success:
        return f"{message}. Source registered as '{source_name}'."
    else:
        return message

@mcp.tool()
def create_isosurface(value: float, field: str = None) -> str:
    """
    Create an isosurface visualization of the active source.
    
    Args:
        value: Isovalue
        field: Optional field name to contour by
    
    Returns:
        Status message
    """
    success, message, contour_obj, contour_name = pv_manager.create_isosurface(value, field)
    if success:
        # Return a user-friendly message that also includes the name
        return f"{message}. Filter registered as '{contour_name}'."
    else:
        return message

@mcp.tool()
def create_slice(origin_x: float = None, origin_y: float = None, origin_z: float = None,
                 normal_x: float = 0, normal_y: float = 0, normal_z: float = 1) -> str:
    """
    Create a slice through the loaded volume data.
    
    Args:
        origin_x, origin_y, origin_z: Coordinates for the slice plane's origin. If None,
            defaults to the data set's center.
        normal_x, normal_y, normal_z: Normal vector for the slice plane (default [0, 0, 1]).
    
    Returns:
        A string message containing success/failure details, plus the pipeline name.
    """
    success, message, slice_filter, slice_name = pv_manager.create_slice(
        origin_x,
        origin_y,
        origin_z,
        normal_x,
        normal_y,
        normal_z
    )

    # Return either an error message or a success message including the slice's name
    return message if success else f"Error creating slice: {message}"

@mcp.tool()
def toggle_volume_rendering(enable: bool = True) -> str:
    """
    Toggle the visibility of volume rendering for the active source.
    
    Args:
        enable (bool): Whether to show (True) or hide (False) volume rendering.
                      If True, shows volume rendering (switching to 'Volume' representation if needed).
                      If False, hides the volume but preserves the volume representation settings.
    
    Returns:
        Status message
    """
       
    success, message, source_name = pv_manager.create_volume_rendering(enable)
    if success:
        # Return a user-friendly message that also includes the name
        return f"{message}. Source registered as '{source_name}'."
    else:
        return message

@mcp.tool()
def toggle_visibility(enable: bool = True) -> str:
    """
    Toggle the visibility for the active source.
    
    Args:
        enable (bool): Whether to show (True) or hide (False) the active source.
                      If True, makes the active source visible.
                      If False, hides the active source but preserves the representation settings.
    
    Returns:
        Status message
    """
       
    success, message, source_name = pv_manager.toggle_visibility(enable)
    if success:
        # Return a user-friendly message that also includes the name
        return f"{message}. Source registered as '{source_name}'."
    else:
        return message


@mcp.tool()
def set_active_source(name: str) -> str:
    """
    Set the active pipeline object by its name.

    Usage:
      set_active_source("Contour1")

    Returns a status message.
    """
    success, message = pv_manager.set_active_source(name)
    return message

@mcp.tool()
def get_active_source_names_by_type(source_type: str = None) -> str:
    """
    Get a list of source names filtered by their type.

    Args:
        source_type (str, optional): Filter sources by type (e.g., 'Sphere', 'Contour', etc.).
                                  If None, returns all sources.

    Returns:
        A string message containing the source names or error message.
    """
    success, message, source_names = pv_manager.get_active_source_names_by_type(source_type)
    
    if success and source_names:
        sources_list = "\n- ".join(source_names)
        result = f"{message}:\n- {sources_list}"
        return result
    else:
        return message

# @mcp.tool()
# def edit_volume_opacity(field_name: str, opacity_points: list) -> str:
#     """
#     Edit ONLY the opacity transfer function for the specified field,
#     ensuring we pass only (value, alpha) pairs.

#     [Tips: only needed by volume rendering particularly finetuning the result, likely not needed when the color is ideal, usually the lower value should always have lower opacity]

#     Args:
#         field_name (str): The data array (field) name whose opacity we're adjusting.
#         opacity_points (list of [value, alpha] pairs):
#             Example: [[0.0, 0.0], [50.0, 0.3], [100.0, 1.0]]

#     Returns:
#         A status message (success or error)
#     """
#     success, message = pv_manager.edit_volume_opacity(field_name, opacity_points)
#     return message

# Compatible with OpenAI tool using
@mcp.tool()
def edit_volume_opacity(field_name: str, opacity_points: list[dict[str, float]]) -> str:
    """
    Edit ONLY the opacity transfer function for the specified field.

    Args:
        field_name (str): The scalar field to modify.
        opacity_points (list): A list of dicts like:
            [{"value": 0.0, "alpha": 0.0}, {"value": 50.0, "alpha": 0.3}]

    Returns:
        A status message (success or error)
    """
    formatted_points = [[pt["value"], pt["alpha"]] for pt in opacity_points]
    success, message = pv_manager.edit_volume_opacity(field_name, formatted_points)
    return message

# @mcp.tool()
# def set_color_map(field_name: str, color_points: list) -> str:
#     """
#     Sets the color transfer function for the specified field.

#     [Tips: only volume rendering should be using the set_color_map function, the lower values range corresponds to lower density objects, whereas higher values indicate high physical density. When design the color mapping try to assess the object of interest's density first from the default colormap (low value assigned to blue, high value assigned to red) and re-assign customized color accordingly, the order of the color may need to be adjust based on the rendering result. The more solid object should have higher density (!high value range). And a screen_shot should always be taken once this function is called to assess how to adjust the color_map again.]

#     Args:
#         field_name (str): The name of the field/array (as it appears in ParaView).
#         color_points (list of [value, [r, g, b]]):
#             e.g., [[0.0, [0.0, 0.0, 1.0]], [50.0, [0.0, 1.0, 0.0]], [100.0, [1.0, 0.0, 0.0]]]
#             Each element is (value, (r, g, b)) with r,g,b in [0,1].

#     Returns:
#         A status message as a string (e.g., success or error).
#     """
#     success, message = pv_manager.set_color_map(field_name, color_points)
#     return message

# Compatible with OpenAI tool using
@mcp.tool()
def set_color_map(field_name: str, color_points: list[dict]) -> str:
    """
    Sets the color transfer function for the specified field.

    [Tips: only volume rendering should be using the set_color_map function, the lower values range corresponds to lower density objects, whereas higher values indicate high physical density. When design the color mapping try to assess the object of interest's density first from the default colormap (low value assigned to blue, high value assigned to red) and re-assign customized color accordingly, the order of the color may need to be adjust based on the rendering result. The more solid object should have higher density (!high value range). And a screen_shot should always be taken once this function is called to assess how to adjust the color_map again.]

    Args:
        field_name (str): The name of the field/array (as it appears in ParaView).
        color_points (list of dicts): Each element should be a dict:
            {"value": float, "rgb": [r, g, b]} where r,g,b ∈ [0,1].

            Example:
            [
                {"value": 0.0, "rgb": [0.0, 0.0, 1.0]},
                {"value": 50.0, "rgb": [0.0, 1.0, 0.0]},
                {"value": 100.0, "rgb": [1.0, 0.0, 0.0]}
            ]

    Returns:
        A status message (success or error).
    """
    # Transform color_points to expected internal format: list[tuple[float, tuple[float, float, float]]]
    try:
        formatted_points = [(pt["value"], tuple(pt["rgb"])) for pt in color_points]
    except Exception as e:
        return f"Invalid format for color_points: {e}"

    success, message = pv_manager.set_color_map(field_name, formatted_points)
    return message


@mcp.tool()
def color_by(field: str, component: int = -1, range_min: float | None = None, range_max: float | None = None) -> str:
    """
    Color the active visualization by a specific field.
    This function first checks if the active source can be colored by fields
    (i.e., it's a dataset with arrays) before attempting to apply colors.
    [tips] Volume rendering should not use this function 

    Args:
        field: Field name to color by
        component: Component to color by (-1 for magnitude)
        range_min: Value shown with the lowest color (default: data minimum)
        range_max: Value shown with the highest color (default: data maximum); set both when outliers make the default range useless
    
    Returns:
        Status message including the color range
    """
    success, message = pv_manager.color_by(field, component, range_min, range_max)
    return message

@mcp.tool()
def compute_surface_area() -> str:
    """
    Compute the surface area of the currently active dataset.
    NOTE: Must be a surface mesh or 'Area' array won't exist.
    """
    success, message, area_value = pv_manager.compute_surface_area()
    return message

# @mcp.tool()
# def set_color_map_preset(preset_name: str) -> str:
#     """
#     Set the color map (lookup table) for the current visualization.
#     [tips: this should only be call at the beginning of the volume rendering]

#     Args:
#         preset_name: Name of the color map preset (e.g., "Rainbow", "Cool to Warm", "viridis")
    
#     Returns:
#         Status message
#     """
#     success, message = pv_manager.set_color_map(preset_name)
#     return message

@mcp.tool()
def set_representation_type(rep_type: str) -> str:
    """
    Set the representation type for the active source.
    
    [Tips: This function should not be used for volume rendering]

    Args:
        rep_type: Representation type (Surface, Wireframe, Points, etc.)
    
    Returns:
        Status message
    """
    success, message = pv_manager.set_representation_type(rep_type)
    return message

@mcp.tool()
def get_pipeline() -> str:
    """
    Get the current pipeline structure.
    
    Returns:
        Description of the current pipeline
    """
    success, message = pv_manager.get_pipeline()
    return message

@mcp.tool()
def get_available_arrays() -> str:
    """
    Get a list of available arrays in the active source.

    [tips: normally volume rendering would not require this information]
    
    Returns:
        List of available arrays
    """
    success, message = pv_manager.get_available_arrays()
    return message

@mcp.tool()
def create_streamline(seed_point_number: int, vector_field: str = None,
                     integration_direction: str = "BOTH", max_steps: int = 1000,
                     initial_step: float = 0.1, maximum_step: float = 50.0) -> str:
    """
    Create streamlines from the loaded vector volume using the StreamTracer filter.
    This function automatically generates seed points based on the data bounds.
    
    Args:
        seed_point_number (int): The number of seed points to automatically generate.
        vector_field (str, optional): The name of the vector field to use for tracing. 
                                    If None, the first vector field will be chosen automatically.
        integration_direction (str): Integration direction ("FORWARD", "BACKWARD", or "BOTH"; default: "BOTH").
        max_steps (int): Maximum number of integration steps (default: 1000).
        initial_step (float): Initial integration step length (default: 0.1).
        maximum_step (float): Maximum streamline length (default: 50.0).
        
    Returns:
        str: Status message indicating whether the streamline was successfully created.
    """
    # Call the stream tracer creation method in your ParaViewManager
    success, message, streamline, tube_name = pv_manager.create_stream_tracer(
        vector_field=vector_field,
        base_source=None,  # Use the active source
        point_center=None,  # Auto-calculate the center
        integration_direction=integration_direction,
        initial_step_length=initial_step,
        maximum_stream_length=maximum_step,
        number_of_streamlines=seed_point_number
    )
    
    if success:
        return f"{message} Tube registered as '{tube_name}'."
    else:
        return message

@mcp.tool(structured_output=False)
def get_screenshot(width: int = 1280, height: int = 720) -> Image | str:
    """
    Capture a screenshot of the current view and display it in chat.

    Args:
        width: Image width in pixels (default: 1280)
        height: Image height in pixels (default: 720)

    Returns:
        The PNG image, or an error message
    """
    success, message, img_path = pv_manager.get_screenshot(width, height)

    if not success:
        return message
    try:
        return Image(data=Path(img_path).read_bytes(), format="png")
    finally:
        Path(img_path).unlink(missing_ok=True)

@mcp.tool(structured_output=False)
def render_animation(start_index: int = 0, end_index: int = -1, max_frames: int = 30, width: int = 800, height: int = 450) -> list[Image | str] | str:
    """
    Render the current view over time and show it to the user as an animation
    with play/pause controls. Set up data, coloring and camera first; they stay
    the same in every frame. You receive only a few of the frames.

    Args:
        start_index: First time step index (default: 0)
        end_index: Last time step index; -1 is the last time step (default: -1)
        max_frames: Maximum number of evenly spaced frames; each takes about a second (default: 30)
        width: Frame width in pixels (default: 800)
        height: Frame height in pixels (default: 450)

    Returns:
        A summary followed by the PNG frames, or an error message
    """
    success, message, frames = pv_manager.render_animation(start_index, end_index, max_frames, width, height)
    if not success:
        return message
    return [message, *(Image(data=frame, format="png") for frame in frames)]

def _run_paths(runs: list[str]) -> list[str]:
    """Resolve run directories (or files in them) within the data directories."""
    paths = []
    for run in runs:
        success, message, resolved = pv_manager._resolve_data_path(run)
        if not success:
            raise ValueError(message)
        paths.append(str(resolved))
    return paths

def _series_resource(payload: dict) -> EmbeddedResource:
    return EmbeddedResource(
        type="resource",
        resource=BlobResourceContents(
            uri="series://time-series",
            mimeType=time_series.SERIES_MEDIA_TYPE,
            blob=base64.b64encode(time_series.payload_bytes(payload)).decode(),
        ),
    )

@mcp.tool()
def list_time_series(run: str) -> str:
    """
    List the time series of an LS-DYNA run (binout history output): categories
    such as glstat (global energies), matsum (part energies), nodout (nodes),
    secforc (cross sections), rcforc (contacts), rwforc (rigid walls), with
    their variables and entities (parts, nodes, sections, ...).

    Args:
        run: Run directory or a file in it, e.g. a00a00/Exp_088 or a00a00/Exp_088/d3plot

    Returns:
        One line per category with its variables and entities
    """
    try:
        return time_series.list_time_series(_run_paths([run])[0])
    except Exception as e:
        return f"Error listing time series: {e}"

@mcp.tool(structured_output=False)
def get_time_series(runs: list[str], category: str, variables: list[str], entities: list[str] | None = None, title: str | None = None) -> list | str:
    """
    Show time series of one or more runs to the user as an interactive chart.
    You receive a summary (min, max, peak time, final value per series), not
    the data. Several runs are drawn on the time axis of the first one, which
    compares design variants.

    Args:
        runs: Run directories or files in them, e.g. ["a00a00/Exp_088", "a00a00/Exp_308"]
        category: Category from list_time_series, e.g. "glstat", "secforc", "rwforc/forces"
        variables: Variables of the category, e.g. ["total_force"] or ["kinetic_energy", "internal_energy"]
        entities: Entity ids or names (or parts of names) to include, e.g. ["c0_l"]; default all
        title: Chart title (default: category and variables)

    Returns:
        A summary and the chart data, or an error message
    """
    try:
        summary, payload = time_series.get_time_series(_run_paths(runs), category, variables, entities, title)
    except Exception as e:
        return f"Error reading time series: {e}"
    return [summary, _series_resource(payload)]

@mcp.tool(structured_output=False)
def compare_runs(file_paths: list[str], field: str, range_min: float | None = None, range_max: float | None = None, time_index: int = -1, frames: int = 1, width: int = 1200, height: int = 500) -> list[Image | str] | str:
    """
    Show 2 to 4 runs side by side with the same coloring, color range, camera
    and time, e.g. design variants of a crash simulation. The camera of the
    current view is used when data is loaded, so set it up first. With frames
    above 1 the comparison is shown as an animation (about a second per frame).

    Args:
        file_paths: Data files of the runs, e.g. ["a00a00/Exp_088/d3plot", "a00a00/Exp_308/d3plot"]
        field: Array to color by, e.g. "EffPlastStrn"
        range_min: Value shown with the lowest color (default: data minimum)
        range_max: Value shown with the highest color (default: data maximum)
        time_index: Time step of a single image; -1 is the last (default: -1)
        frames: Number of evenly spaced time steps; more than 1 animates (default: 1)
        width: Image width in pixels (default: 1200)
        height: Image height in pixels (default: 500)

    Returns:
        A summary followed by the PNG image(s), or an error message
    """
    success, message, images = pv_manager.compare_runs(file_paths, field, range_min, range_max, time_index, frames, width, height)
    if not success:
        return message
    return [message, *(Image(data=image, format="png") for image in images)]

@mcp.tool()
def get_data_info() -> str:
    """
    Describe the active data at the current time: point and cell counts,
    bounds, and min/max/99th percentile of every array. Use it to choose
    probe locations and color ranges.

    Returns:
        Description of the data
    """
    success, message = pv_manager.get_data_info()
    return message

@mcp.tool(structured_output=False)
def sample_over_line(point1: list[float], point2: list[float], field: str, resolution: int = 200, radius: float = 50.0) -> list | str:
    """
    Show the user a chart of an array along a straight line through the
    active data at the current time, e.g. the deflection along the vehicle.
    Each sample takes the nearest point (or cell center) within radius, so
    thin shell structures near the line are found.

    Args:
        point1: Start point [x, y, z]
        point2: End point [x, y, z]
        field: Array to sample, e.g. "Deflection" (vectors become magnitudes)
        resolution: Number of samples (default: 200)
        radius: Search radius around each sample in model units (default: 50)

    Returns:
        A summary and the chart data, or an error message
    """
    success, message, payload = pv_manager.sample_over_line(point1, point2, field, resolution, radius)
    if not success:
        return message
    return [message, _series_resource(payload)]

@mcp.tool(structured_output=False)
def probe_over_time(field: str, point: list[float] | None = None, start_index: int = 0, end_index: int = -1, max_samples: int = 60) -> list | str:
    """
    Show the user a chart of an array over time at one node of the loaded
    data. The node moves with the material. It is the node nearest to point,
    or without a point the node where the array is largest at the current
    time. Cell arrays are averaged over the cells around the node. Each time
    step takes up to a second.

    Args:
        field: Array to record, e.g. "Deflection" or "EffPlastStrn"
        point: [x, y, z] near the node at the current time (default: location of the maximum)
        start_index: First time step index (default: 0)
        end_index: Last time step index; -1 is the last (default: -1)
        max_samples: Maximum number of evenly spaced time steps (default: 60)

    Returns:
        A summary and the chart data, or an error message
    """
    success, message, payload = pv_manager.probe_over_time(field, point, start_index, end_index, max_samples)
    if not success:
        return message
    return [message, _series_resource(payload)]

@mcp.tool()
def list_data_files(pattern: str = "**/*", limit: int = 200) -> str:
    """
    List files in the data directories that can be passed to load_data.

    Args:
        pattern: Glob pattern relative to each data directory, e.g. "**/d3plot" (default: all files)
        limit: Maximum number of files to list (default: 200)

    Returns:
        Matching file paths, one per line
    """
    success, message, paths = pv_manager.list_data_files(pattern, limit)
    if not success or not paths:
        return message
    return f"{message}:\n" + "\n".join(paths)

@mcp.tool()
def get_timesteps() -> str:
    """
    Describe the time steps of the loaded data (count, range and current time).

    Returns:
        Status message
    """
    success, message = pv_manager.get_timesteps()
    return message

@mcp.tool()
def set_time(index: int | None = None, value: float | None = None) -> str:
    """
    Show the loaded data at one time step.

    Args:
        index: Time step index; -1 is the last time step
        value: Time value; the nearest time step is used when no index is given

    Returns:
        Status message
    """
    success, message = pv_manager.set_time(index, value)
    return message

@mcp.tool()
def reset_session() -> str:
    """
    Remove all loaded data and filters to start a new analysis.

    Returns:
        Status message
    """
    success, message = pv_manager.reset_session()
    return message
    
@mcp.tool()
def rotate_camera(azimuth: float = 30.0, elevation: float = 0.0) -> str:
    """
    Rotate the camera by specified angles.
    
    Args:
        azimuth: Rotation around vertical axis in degrees
        elevation: Rotation around horizontal axis in degrees
    
    Returns:
        Status message
    """
    success, message = pv_manager.rotate_camera(azimuth, elevation)
    return message

@mcp.tool()
def reset_camera() -> str:
    """
    Reset the camera to show all data.
    
    Returns:
        Status message
    """
    success, message = pv_manager.reset_camera()
    return message

# @mcp.tool()
# def plot_over_line(point1: list = None, point2: list = None, resolution: int = 100) -> str:
#     """
#     Create a 'Plot Over Line' filter to sample data along a line between two points.

#     Args:
#         point1 (list, optional): The [x, y, z] coordinates of the start point. If None, will use data bounds.
#         point2 (list, optional): The [x, y, z] coordinates of the end point. If None, will use data bounds.
#         resolution (int, optional): Number of sample points along the line (default: 100).

#     Returns:
#         Status message
#     """
#     success, message, plot_filter = pv_manager.plot_over_line(point1, point2, resolution)
#     return message

# Compatible with OpenAI tool using
@mcp.tool()
def plot_over_line(point1: list[float] = None, point2: list[float] = None, resolution: int = 100) -> str:
    """
    Create a 'Plot Over Line' filter to sample data along a line between two points.

    Args:
        point1 (list of float): The [x, y, z] coordinates of the start point. If None, will use data bounds.
        point2 (list of float): The [x, y, z] coordinates of the end point. If None, will use data bounds.
        resolution (int): Number of sample points along the line (default: 100).

    Returns:
        Status message
    """
    success, message, plot_filter = pv_manager.plot_over_line(point1, point2, resolution)
    return message

@mcp.tool()
def warp_by_vector(vector_field: str = None, scale_factor: float = 1.0) -> str:
    """
    Apply the 'Warp By Vector' filter to the active source.

    Args:
        vector_field (str, optional): The name of the vector field to use for warping. If None, the first available vector field will be used.
        scale_factor (float, optional): The scale factor for the warp (default: 1.0).

    Returns:
        Status message
    """
    success, message, warp_filter = pv_manager.warp_by_vector(vector_field, scale_factor)
    return message

@mcp.tool()
def list_commands() -> str:
    """
    List all available commands in this ParaView MCP server.
    
    Returns:
        List of available commands
    """
    commands = [
        "load_data: Load data from a file",
        "create_source: Create a geometric source (Sphere, Cone, etc.)",
        "create_isosurface: Create an isosurface visualization",
        "create_slice: Create a slice through the data",
        "toggle_volume_rendering: Enable or disable volume rendering",
	    "toggle_visibility: Enable or disable visibility for the active source",
        "set_active_source: Set the active pipeline object by name",
        "get_active_source_names_by_type: Get a list of sources filtered by type",
        "color_by: Color the visualization by a field",
        # "set_color_map_preset: Set the color map preset",
        "set_representation_type: Set the representation type (Surface, Wireframe, etc.)",
        "edit_volume_opacity: Edit the opacity transfer function",
        "get_pipeline: Get the current pipeline structure",
        "get_available_arrays: Get available data arrays",
        "create_streamline: Create stream line visualization",
        "compute_surface_area: Compute the surface area of the active surface",
        "save_contour_as_stl: Save the active surface as STL",
        "get_screenshot: Capture a screenshot and display it in chat",
        "render_animation: Show the current view over time as an animation",
        "compare_runs: Show 2 to 4 runs side by side",
        "list_data_files: List files in the data directories",
        "get_data_info: Describe bounds and array ranges of the active data",
        "sample_over_line: Chart an array along a line",
        "probe_over_time: Chart an array over time at a node",
        "list_time_series: List the binout time series of a run",
        "get_time_series: Chart time series of one or more runs",
        "get_timesteps: Describe the time steps of the loaded data",
        "set_time: Show the data at a time step",
        "reset_session: Remove all loaded data and filters",
        "rotate_camera: Rotate the camera view",
        "reset_camera: Reset the camera to show all data",
        "plot_line: Plot a line through the data",
        "warp_by_vector: Warp the active source by a vector field",
    ]
    
    return "Available ParaView commands:\n\n" + "\n".join(commands)


def main():
    parser = argparse.ArgumentParser(description="ParaView External MCP Server")
    parser.add_argument("--server", type=str, default="localhost", help="ParaView server hostname, or 'builtin' to render in this process without a GUI (default: localhost)")
    parser.add_argument("--port", type=int, default=11111, help="ParaView server port (default: 11111)")
    parser.add_argument("--paraview_package_path", type=str, help="Path to the ParaView Python package", default=None)
    parser.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio", help="MCP transport (default: stdio)")
    parser.add_argument("--http-host", type=str, default="127.0.0.1", help="Address to listen on with streamable-http (default: 127.0.0.1)")
    parser.add_argument("--http-port", type=int, default=8765, help="Port to listen on with streamable-http (default: 8765)")
    parser.add_argument("--allowed-host", action="append", default=[], help="Host header accepted with streamable-http, e.g. 'paraview-mcp:*' (repeatable; default: only localhost when listening on localhost)")
    parser.add_argument("--data-root", action="append", default=[], help="Directory data may be loaded from (repeatable; default: any path)")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for written files (default: next to the loaded data)")
    parser.add_argument("--probe-timeout", type=float, default=120.0, help="Seconds a file may take to open in a probe process before it is rejected; 0 disables probing (default: 120)")

    args = parser.parse_args()

    # Add the ParaView package path to sys.path
    if args.paraview_package_path:
        sys.path.append(args.paraview_package_path)

    pv_manager.configure(
        data_roots=args.data_root,
        output_dir=args.output_dir,
        probe_timeout=args.probe_timeout,
    )

    # Connect to ParaView
    pv_manager.connect(args.server, args.port)

    if args.transport == "streamable-http":
        mcp.settings.host = args.http_host
        mcp.settings.port = args.http_port
        if args.allowed_host:
            mcp.settings.transport_security = TransportSecuritySettings(
                enable_dns_rebinding_protection=True,
                allowed_hosts=args.allowed_host,
            )
        elif args.http_host not in ("127.0.0.1", "localhost", "::1"):
            # Reached by other containers under a service name; Host headers vary.
            mcp.settings.transport_security = TransportSecuritySettings(
                enable_dns_rebinding_protection=False,
            )

    # Run the MCP server
    try:
        logger.info("Starting ParaView External MCP Server")
        logger.info(f"ParaView server: {args.server}:{args.port}")
        # logger.info("Default prompt enabled: Claude will call one function per reply")

        # Run the MCP server
        mcp.run(transport=args.transport)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Error running MCP server: {str(e)}")

if __name__ == "__main__":
    main()
