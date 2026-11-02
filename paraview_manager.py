"""
ParaView MCP Server

This class encapsulates paraview.simple API to expose a higher-level API that is compatible with LLM access/control.
"""

import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from paraview.simple import *

# Opens a file in a separate process. Some readers never return on broken
# input (e.g. LS-DYNA d3plot files whose state files are missing), which would
# block the MCP server forever if attempted in-process.
_PROBE_SCRIPT = """
import sys
from paraview.simple import OpenDataFile
reader = OpenDataFile(sys.argv[1])
if reader is None:
    sys.exit(2)
reader.UpdatePipelineInformation()
"""

class ParaViewManager:
    """
    Encapsulates all ParaView-specific functionality.
    This class handles the interaction with ParaView and provides
    a clean interface for the MCP server.
    """

    def __init__(self):
        """Initialize the ParaView manager"""
        self.connection = None
        self.logger = logging.getLogger("paraview_manager")
        # This will always hold the originally loaded data source,
        # which is needed for operations like volume rendering.
        self.original_source = None
        self._data_folder = ""
        # True when connected to a pvserver shared with a ParaView GUI.
        self.collaboration = False
        # Directories data may be loaded from; empty allows any path.
        self.data_roots = []
        # Directory for written files; None writes next to the loaded data.
        self.output_dir = None
        # Seconds a file may take to open in a probe process; 0 disables probing.
        self.probe_timeout = 120.0

    def configure(self, data_roots=(), output_dir=None, probe_timeout=120.0):
        """
        Restrict data access and set where files are written.

        Args:
            data_roots: Directories data may be loaded from (empty allows any path)
            output_dir: Directory for written files (None writes next to the data)
            probe_timeout: Seconds a file may take to open before it is rejected
        """
        self.data_roots = [Path(root).resolve() for root in data_roots]
        self.output_dir = Path(output_dir).resolve() if output_dir else None
        self.probe_timeout = probe_timeout

    def _resolve_data_path(self, file_path):
        """
        Resolve a path and check it lies within a data root.

        Returns:
            tuple: (success, message, resolved_path)
        """
        path = Path(file_path).expanduser()
        if not path.is_absolute() and self.data_roots:
            path = self.data_roots[0] / path
        path = path.resolve()
        if self.data_roots and not any(
            path == root or root in path.parents for root in self.data_roots
        ):
            roots = ", ".join(str(root) for root in self.data_roots)
            return False, f"Error: {file_path} is outside the data directories ({roots}).", None
        if not path.exists():
            return False, f"Error: {path} does not exist.", None
        return True, "", path

    def _probe_readable(self, path):
        """
        Open a file in a subprocess to reject files a reader cannot handle.

        Returns:
            tuple: (success, message)
        """
        if not self.probe_timeout:
            return True, ""
        try:
            result = subprocess.run(
                [sys.executable, "-c", _PROBE_SCRIPT, str(path)],
                capture_output=True,
                timeout=self.probe_timeout,
            )
        except subprocess.TimeoutExpired:
            return False, (
                f"Error: {path} could not be opened within {self.probe_timeout:.0f}s. "
                "The file is probably truncated or incomplete (for LS-DYNA, check "
                "that the d3plot01, d3plot02, ... state files are present)."
            )
        if result.returncode != 0:
            return False, f"Error: ParaView has no reader that can open {path}."
        return True, ""
    
    def _get_source_name(self, proxy):
        """
        Get the name (registered name) of a source proxy.
        
        Args:
            proxy: The source proxy object.
            
        Returns:
            str: The name of the proxy, or empty string if not found.
        """
        try:
            from paraview.simple import GetSources
            
            if proxy is None:
                return ""
                
            sources_dict = GetSources()
            for (key, src_proxy) in sources_dict.items():
                if src_proxy == proxy:
                    return key[0]  # Return the first element (name) of the key tuple
            
            return ""  # Return empty string if proxy not found
        except Exception as e:
            self.logger.error(f"Error getting source name: {str(e)}")
            return ""
    
    def connect(self, server_url="localhost", port=11111):
        """
        Connect to a running ParaView server
        
        Args:
            server_url: Server hostname (default: localhost)
            port: Server port (default: 11111)
            
        Returns:
            bool: Success status
        """
        try:
            # Import the paraview.simple module
            import importlib.util

            if server_url == "builtin":
                # Render in this process; no ParaView GUI shares the session.
                self.logger.info("Using the built-in ParaView session")
                return True

            if importlib.util.find_spec("paraview.simple") is not None:
                from paraview.simple import Connect, GetActiveView
                
                # Connect to the existing ParaView server
                full_server_url = f"{server_url}:{port}" if port else server_url
                self.logger.info(f"Connecting to ParaView at {full_server_url}")
                self.connection = Connect(full_server_url)
                self.collaboration = True
                
                # Get the active view to confirm connection
                view = GetActiveView()
                
                self.logger.info("Successfully connected to ParaView")
                return True
            else:
                self.logger.warning("paraview.simple module not found. Running in simulation mode.")
                return False
        except Exception as e:
            self.logger.error(f"Failed to connect to ParaView: {str(e)}")
            return False
    
    def load_data(self, file_path):
        """
        Load data from a file into ParaView
        
        Args:
            file_path: Path to the data file
            
        Returns:
            tuple: (success, message, reader, source_name)
        """
        try:
            import os
            from paraview.simple import OpenDataFile, Show, GetActiveViewOrCreate

            success, message, resolved = self._resolve_data_path(file_path)
            if not success:
                return False, message, None, ""
            file_path = str(resolved)
            success, message = self._probe_readable(file_path)
            if not success:
                return False, message, None, ""

            # Record the directory of the loaded file so we can re-use it.
            self._data_folder = os.path.dirname(file_path)

            # Get file extension
            _, file_extension = os.path.splitext(file_path)
            file_extension = file_extension.lower()
            file_name = os.path.basename(file_path)

            # Special handling for raw volume files
            if file_extension == '.raw':
                reader = self._configure_raw_reader(file_path, file_name)
            else:
                # Standard loading for other file types
                reader = OpenDataFile(file_path)
            
            if not reader:
                return False, f"Failed to load data from {file_path}", None, ""
            
            # Show in the active view
            view = GetActiveViewOrCreate("RenderView")
            display = Show(reader, view)
            display.ScaleFactor = 0.5
            view.ResetCamera(False)
            # Save the loaded reader as the original data source
            self.original_source = reader
            
            # Get the source name using the helper function
            source_name = self._get_source_name(reader)
            
            return True, f"Successfully loaded data from {file_path}", reader, source_name
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            return False, f"Error loading data: {str(e)}", None, ""

    
    def _configure_raw_reader(self, file_path, file_name):
        """
        Configure a reader for RAW volume files
        
        Args:
            file_path: Path to the RAW file
            file_name: Name of the file
            
        Returns:
            reader: Configured reader object
        """
        from paraview.simple import OpenDataFile
        
        # Try to parse dimensions and data type from filename
        # Expected format: name_XxYxZ_datatype.raw (e.g., foot_256x256x256_uint8.raw)
        dimensions_match = re.search(r'(\d+)x(\d+)x(\d+)', file_name)
        datatype_match = re.search(r'_(uint8|uint16|int8|int16|float32|float64)', file_name.lower())
        
        # Load the raw file
        reader = OpenDataFile(file_path)
        if not reader:
            return None
        
        # Set reader properties based on filename
        if dimensions_match:
            dim_x = int(dimensions_match.group(1))
            dim_y = int(dimensions_match.group(2))
            dim_z = int(dimensions_match.group(3))
            reader.DataExtent = [0, dim_x-1, 0, dim_y-1, 0, dim_z-1]
            reader.FileDimensionality = 3
            self.logger.info(f"Detected dimensions: {dim_x}x{dim_y}x{dim_z}")
        
        if datatype_match:
            datatype = datatype_match.group(1)
            # Map to ParaView data types
            datatype_map = {
                'uint8': 'unsigned char',
                'uint16': 'unsigned short',
                'int8': 'char',
                'int16': 'short',
                'float32': 'float',
                'float64': 'double'
            }
            if datatype in datatype_map:
                reader.DataScalarType = datatype_map[datatype]
                self.logger.info(f"Detected data type: {datatype_map[datatype]}")
        else:
            # Default to unsigned char if not specified
            reader.DataScalarType = 'unsigned char'
        
        # Set other common properties for raw files
        reader.DataByteOrder = 'LittleEndian'  # Default to LittleEndian
        reader.NumberOfScalarComponents = 1    # Default to single component
        
        self.logger.info(f"Configured RAW reader with: ScalarType={reader.DataScalarType}, " +
                         f"ByteOrder={reader.DataByteOrder}, Extent={reader.DataExtent}")
        
        return reader
    
    def save_contour_as_stl(self, stl_filename="contour.stl"):
        """
        Save the active source (e.g. a contour) as an STL file in the same folder
        where the original data was loaded.

        Args:
            stl_filename (str): Name of the STL file to create (defaults to 'contour.stl').

        Returns:
            tuple: (success: bool, message: str, saved_path: str)
        """
        try:
            import os
            from paraview.simple import GetActiveSource, SaveData

            # Ensure we have an active source
            active_source = GetActiveSource()
            if not active_source:
                return False, "Error: No active source to save.", ""

            if self.output_dir is not None:
                self.output_dir.mkdir(parents=True, exist_ok=True)
                full_path = str(self.output_dir / os.path.basename(stl_filename))
            else:
                # Check that we have a recorded data folder
                if not hasattr(self, "_data_folder") or not self._data_folder:
                    return False, (
                        "Error: No data folder known. "
                        "Did you load data first before saving?"
                    ), ""

                # Compose the full path in the same folder as the loaded data
                full_path = os.path.join(self._data_folder, stl_filename)

            # Save to STL
            SaveData(full_path, proxy=active_source)
            
            message = f"Saved active source to STL at: {full_path}"
            return True, message, full_path
        except Exception as e:
            self.logger.error(f"Error saving STL: {str(e)}")
            return False, f"Error saving STL: {str(e)}", ""
        
    def create_source(self, source_type):
        """
        Create a new geometric source
        
        Args:
            source_type: Type of source to create (Sphere, Cone, etc.)
            
        Returns:
            tuple: (success, message, source, source_name)
        """
        try:
            from paraview.simple import GetActiveView, Show
            source = None
            source_type = source_type.lower()
            
            if source_type == "sphere":
                from paraview.simple import Sphere
                source = Sphere()
            elif source_type == "cone":
                from paraview.simple import Cone
                source = Cone()
            elif source_type == "cylinder":
                from paraview.simple import Cylinder
                source = Cylinder()
            elif source_type == "plane":
                from paraview.simple import Plane
                source = Plane()
            elif source_type == "box":
                from paraview.simple import Box
                source = Box()
            else:
                return False, f"Unsupported source type: {source_type}", None, ""
            
            view = GetActiveView()
            Show(source, view)
            
            # Get the source name using the helper function
            source_name = self._get_source_name(source)
            
            return True, f"Created {source_type} source", source, source_name
        except Exception as e:
            self.logger.error(f"Error creating source: {str(e)}")
            return False, f"Error creating source: {str(e)}", None, ""
    
    def set_active_source(self, name):
        """
        Set the active pipeline object by matching its registered name. Use this function to set active source so that the computation is applied to the correct objects in paraview object hiearchy. 

        Args:
            name (str): The name of the pipeline object, e.g. "Slice1" or "Contour1".
                        Typically, ParaView registers pipeline objects using this sort of naming.

        Returns:
            tuple: (success: bool, message: str)
        """
        try:
            from paraview.simple import GetSources, SetActiveSource

            sources_dict = GetSources()  # Returns a dict: { (name, ""), proxyObject }, etc.
            if not sources_dict:
                return False, "No sources available in the pipeline."

            # Attempt exact or partial match:
            # Option A: Exact match on the first element of the key
            # Option B: A more flexible approach scanning all source names
            matches = []
            for (source_key, proxy) in sources_dict.items():
                # source_key is typically (registeredName, fileNameOrOtherString)
                if source_key[0] == name:
                    SetActiveSource(proxy)
                    return True, f"Active source set to '{source_key[0]}'"
                # Alternatively, you could allow partial or case-insensitive matches:
                # if name.lower() in source_key[0].lower():
                #     matches.append((source_key[0], proxy))

            return False, f"No source found with the name '{name}'."
        except Exception as e:
            self.logger.error(f"Error in set_active_source: {str(e)}")
            return False, f"Error setting active source: {str(e)}"

    def get_active_source_names_by_type(self, source_type=None):
        """
        Get a list of source names filtered by their type.
        
        Args:
            source_type (str, optional): Filter sources by type (e.g., 'Sphere', 'Contour', etc.).
                                      If None, returns all sources.
        
        Returns:
            tuple: (success: bool, message: str, source_names: list)
        """
        try:
            from paraview.simple import GetSources
            
            sources_dict = GetSources()
            if not sources_dict:
                return True, "No sources available in the pipeline.", []
            
            result_sources = []
            
            for (source_key, proxy) in sources_dict.items():
                proxy_type = proxy.__class__.__name__
                
                # If source_type is None or matches the proxy type, add to results
                if source_type is None or source_type.lower() in proxy_type.lower():
                    result_sources.append(source_key[0])
            
            if not result_sources and source_type:
                message = f"No sources of type '{source_type}' found in the pipeline."
            elif not result_sources:
                message = "No sources found in the pipeline."
            else:
                message = f"Found {len(result_sources)} source(s)" + (f" of type '{source_type}'" if source_type else "")
            
            return True, message, result_sources
            
        except Exception as e:
            self.logger.error(f"Error getting source names by type: {str(e)}")
            return False, f"Error getting source names by type: {str(e)}", []


    def create_isosurface(self, value, field=None):
        """
        Create or update an isosurface visualization of the loaded volume data.
        If an isosurface filter already exists (stored in self.isosurface_filter),
        update its isovalue and contour parameters. Otherwise, create a new filter.

        Args:
            value: Isovalue.
            field: Optional field name to contour by.

        Returns:
            tuple: (success: bool, message: str, contour_proxy, contour_name: str)
        """
        try:
            from paraview.simple import (
                GetActiveView, SetActiveSource, Contour, Show, GetActiveSource
            )

            # Use the originally loaded source if available; fall back to the active source.
            base_source = self.original_source or GetActiveSource()
            if not base_source:
                return False, "Error: No active source. Load data first.", None, ""

            # Determine whether to update an existing isosurface or create a new one.
            if hasattr(self, 'isosurface_filter') and self.isosurface_filter:
                contour = self.isosurface_filter
                contour.Isosurfaces = [value]
                if field:
                    contour.ContourBy = ['POINTS', field]
                message = f"Updated isosurface to value {value}"
            else:
                contour = Contour(Input=base_source)
                contour.Isosurfaces = [value]
                if field:
                    contour.ContourBy = ['POINTS', field]
                self.isosurface_filter = contour
                message = f"Created isosurface at value {value}"

            # Show the contour in the active view
            view = GetActiveView()
            Show(contour, view)

            # Optionally reset active source to the original data
            SetActiveSource(base_source)

            # Get the source name using the helper function
            contour_name = self._get_source_name(contour)

            # Return a 4-tuple including the name
            return True, message, contour, contour_name

        except Exception as e:
            self.logger.error(f"Error creating/updating isosurface: {str(e)}")
            return False, f"Error creating/updating isosurface: {str(e)}", None, ""

    def compute_surface_area(self):
        """
        Compute the surface area of the ACTIVE source.

        IMPORTANT: This assumes the active pipeline object is a surface mesh.
        If the active pipeline is still a volumetric dataset, you won't get
        a valid 'Area' array. For example, you might want to call:
        1) extract_surface()
        2) [SetActiveSource(...) for the extracted surface]
        3) compute_surface_area()

        Returns:
            tuple: (success: bool, message: str, area_value: float)
        """
        try:
            from paraview.simple import GetActiveSource, IntegrateVariables
            import paraview.servermanager as sm

            source = GetActiveSource()
            if not source:
                return False, "Error: No active source. Load data first.", 0.0

            # IntegrateVariables on a surface dataset yields an 'Area' array
            integrate_filter = IntegrateVariables(Input=source)
            integrate_filter.UpdatePipeline()

            # Fetch integrated results
            integrated_data = sm.Fetch(integrate_filter)
            if not integrated_data:
                return False, "Error: Could not fetch integrated data from server.", 0.0

            # Look for 'Area' array in CellData
            area_array = integrated_data.GetCellData().GetArray("Area")
            if not area_array:
                return False, (
                    "No 'Area' array found. Are you sure this is a surface dataset?"
                ), 0.0

            area_value = area_array.GetValue(0)
            return True, f"Computed surface area: {area_value}", area_value

        except Exception as e:
            self.logger.error(f"Error computing surface area: {str(e)}")
            return False, f"Error computing surface area: {str(e)}", 0.0

            # The integrated filter typically stores one value (the total area) in index 0
            total_area = area_array.GetValue(0)

            return (True, "Successfully computed surface area.", total_area)

        except Exception as e:
            self.logger.error(f"Error computing surface area: {str(e)}")
            return (False, f"Error computing surface area: {str(e)}", None)
        
    def create_slice(self, origin_x=None, origin_y=None, origin_z=None,
                 normal_x=0, normal_y=0, normal_z=1):
        """
        Create a slice through the loaded volume data.

        Args:
            origin_x, origin_y, origin_z: Coordinates for slice origin (default: center of dataset).
                                        If None, it uses the dataset's center.
            normal_x, normal_y, normal_z: Normal of the slice plane (default: [0, 0, 1]).

        Returns:
            tuple: (success: bool, message: str, slice_filter, slice_name: str)
        """
        try:
            from paraview.simple import (
                GetActiveView, SetActiveSource, Slice, Show, GetActiveSource
            )

            base_source = self.original_source or GetActiveSource()
            if not base_source:
                return False, "Error: No active source. Load data first.", None, None

            # If origin is unspecified, use the center of the dataset
            if origin_x is not None and origin_y is not None and origin_z is not None:
                origin = [origin_x, origin_y, origin_z]
            else:
                info = base_source.GetDataInformation()
                bounds = info.GetBounds()
                origin = [
                    (bounds[0] + bounds[1]) / 2,
                    (bounds[2] + bounds[3]) / 2,
                    (bounds[4] + bounds[5]) / 2
                ]

            normal = [normal_x, normal_y, normal_z]

            # Create and configure the slice filter
            slice_filter = Slice(Input=base_source)
            slice_filter.SliceType = 'Plane'
            slice_filter.SliceType.Origin = origin
            slice_filter.SliceType.Normal = normal

            # Show the new slice in the view
            view = GetActiveView()
            Show(slice_filter, view)

            # (Optional) reset the active source to the original volume
            SetActiveSource(base_source)

            # Get the source name using the helper function
            slice_name = self._get_source_name(slice_filter)

            message = (
                f"Created slice with origin {origin} and normal {normal}. "
                f"Slice name is: {slice_name}"
            )
            return True, message, slice_filter, slice_name

        except Exception as e:
            self.logger.error(f"Error creating slice: {str(e)}")
            return False, f"Error creating slice: {str(e)}", None, None

        
    def create_volume_rendering(self, enable=True):
        """
        Toggle volume rendering for the loaded volume data.
        
        Args:
            enable (bool): Whether to enable (True) or disable (False) volume rendering.
                          If True, shows volume rendering.
                          If False, hides the volume but preserves the volume representation.
        
        Returns:
            tuple: (success, message, source_name)
        """
        try:
            from paraview.simple import GetActiveView, SetActiveSource, GetDisplayProperties

            if not self.original_source:
                return False, "Error: No original data loaded. Load data first.", None

            # Force the original volume data to be active
            SetActiveSource(self.original_source)
            view = GetActiveView()
            display = GetDisplayProperties(self.original_source, view)

            # Get the current representation type
            current_rep = display.GetRepresentationType() if hasattr(display, 'GetRepresentationType') else None
            
            if enable:
                # Switch to Volume representation if not already
                if current_rep != 'Volume':
                    display.SetRepresentationType('Volume')
                # Make sure it's visible
                display.Visibility = 1
                status_message = "Volume rendering enabled"
            else:
                # If currently in Volume mode, make it invisible
                # but don't change the representation type
                if current_rep == 'Volume':
                    display.Visibility = 0
                    status_message = "Volume rendering hidden (representation preserved)"
                else:
                    # If not in Volume mode, just report current state
                    status_message = f"Volume rendering already disabled (current representation: {current_rep})"

            # Get the source name using the helper function
            source_name = self._get_source_name(self.original_source)

            return True, status_message, source_name

        except Exception as e:
            self.logger.error(f"Error toggling volume rendering: {str(e)}")
            return False, f"Error toggling volume rendering: {str(e)}", None

    def toggle_visibility(self, enable=True):
        """
        Toggle visibility for the current source.
        
        Args:
            enable (bool): Whether to enable (True) or disable (False) visibility of the current source.
                          If True, shows the current source.
                          If False, hides the current source.
        
        Returns:
            tuple: (success, message, source_name)
        """
        try:
            from paraview.simple import GetActiveView, SetActiveSource, GetDisplayProperties

            if not GetActiveSource():
                return False, "Error: No data selected. Load data first.", None

            view = GetActiveView()
            display = GetDisplayProperties(GetActiveSource(), view)
            
            if enable:
                display.Visibility = 1
                status_message = "Element was made visibile"
            else:
                display.Visibility = 0
                status_message = "Rendering hidden (representation preserved)"

            # Get the source name using the helper function
            source_name = self._get_source_name(GetActiveSource())

            return True, status_message, source_name

        except Exception as e:
            self.logger.error(f"Error toggling visibility: {str(e)}")
            return False, f"Error toggling visibility: {str(e)}", None
    
    def color_by(self, field, component=-1, range_min=None, range_max=None):
        """
        Color the active visualization by a specific field.
        This function first checks if the active source can be colored by fields
        (i.e., it's a dataset with arrays) before attempting to apply colors.
        
        Args:
            field: Field name to color by.
            component: Component to color by (-1 for magnitude).
            range_min: Value shown with the lowest color (default: data minimum).
            range_max: Value shown with the highest color (default: data maximum).
            
        Returns:
            tuple: (success, message)
        """
        try:
            from paraview.simple import GetActiveSource, GetActiveView, GetDisplayProperties, ColorBy, GetColorTransferFunction, GetOpacityTransferFunction, GetScalarBar
            
            source = GetActiveSource()
            if not source:
                return False, "Error: No active source. Load data first."
            
            view = GetActiveView()
            display = GetDisplayProperties(source, view)
            
            # Check if the current representation type can be colored by arrays
            # Some representations (like 'Outline') cannot be colored by data arrays
            rep_type = display.GetRepresentationType() if hasattr(display, 'GetRepresentationType') else None
            if rep_type in ['Outline', 'Wireframe']:
                return False, f"Error: The current representation type '{rep_type}' cannot be colored by fields. Try changing to 'Surface' or 'Volume' first."
            
            # Get data information directly from the source
            data_info = source.GetDataInformation()
            point_info = data_info.GetPointDataInformation()
            cell_info = data_info.GetCellDataInformation()
            
            # Check if the active source has data arrays
            if (point_info.GetNumberOfArrays() == 0 and 
                cell_info.GetNumberOfArrays() == 0):
                return False, "Error: The active source does not have any data arrays to color by."
            
            # Try to find the requested field
            field_available = False
            field_location = None
            
            # Check point data arrays
            for i in range(point_info.GetNumberOfArrays()):
                array_info = point_info.GetArrayInformation(i)
                if array_info.GetName() == field:
                    ColorBy(display, ('POINTS', field))
                    field_available = True
                    field_location = 'POINTS'
                    break
            
            # Check cell data arrays if not found in point data
            if not field_available:
                for i in range(cell_info.GetNumberOfArrays()):
                    array_info = cell_info.GetArrayInformation(i)
                    if array_info.GetName() == field:
                        ColorBy(display, ('CELLS', field))
                        field_available = True
                        field_location = 'CELLS'
                        break
            
            if not field_available:
                # Build a list of available fields for better error reporting
                available_fields = []
                for i in range(point_info.GetNumberOfArrays()):
                    array_info = point_info.GetArrayInformation(i)
                    available_fields.append(f"{array_info.GetName()} (POINTS)")
                for i in range(cell_info.GetNumberOfArrays()):
                    array_info = cell_info.GetArrayInformation(i)
                    available_fields.append(f"{array_info.GetName()} (CELLS)")
                
                fields_str = ", ".join(available_fields)
                return False, f"Error: Field '{field}' not found. Available fields are: {fields_str}"
            
            # ColorBy's third argument would request a separate color map;
            # the component is chosen on the shared color map instead.
            lut = GetColorTransferFunction(field)
            if component is not None and component >= 0:
                lut.VectorMode = "Component"
                lut.VectorComponent = component
            else:
                lut.VectorMode = "Magnitude"

            # Rescale the color map to show the full data range
            display.RescaleTransferFunctionToDataRange(False, True)
            if range_min is not None or range_max is not None:
                low, high = lut.RGBPoints[0], lut.RGBPoints[-4]
                low = low if range_min is None else range_min
                high = high if range_max is None else range_max
                # Keep the chosen range when the data or time step changes.
                lut.AutomaticRescaleRangeMode = "Never"
                lut.RescaleTransferFunction(low, high)
                GetOpacityTransferFunction(field).RescaleTransferFunction(low, high)
            display.SetScalarBarVisibility(view, True)
            scalar_bar = GetScalarBar(lut, view)
            scalar_bar.WindowLocation = "Upper Right Corner"
            scalar_bar.ScalarBarLength = 0.3
            lut_range = lut.RGBPoints
            return True, (
                f"Colored by field: '{field}' from {field_location}, "
                f"color range {lut_range[0]:g} to {lut_range[-4]:g}"
            )
        except Exception as e:
            self.logger.error(f"Error coloring by field: {str(e)}")
            return False, f"Error coloring by field: {str(e)}"

    
    def set_color_map(self, preset_name="Blue-Red"):
        """
        Set the color map (lookup table) for the current visualization.
        
        Args:
            preset_name: Name of the color map preset.
                        Available presets include (but are not limited to):
                        - Blue-Red
                        - Cool to Warm
                        - Viridis
                        - Plasma
                        - Magma
                        - Inferno
                        - Rainbow
                        - Grayscale
                        
        Returns:
            tuple: (success, message)
        """
        try:
            from paraview.simple import GetActiveSource, GetActiveView, GetDisplayProperties, ApplyPreset
            source = GetActiveSource()
            if not source:
                return False, "Error: No active source. Load data first."
            
            view = GetActiveView()
            display = GetDisplayProperties(source, view)
            
            color_tf = display.LookupTable
            if not color_tf:
                return False, "Error: No active color transfer function"
            
            # Apply the requested preset to the color transfer function.
            ApplyPreset(color_tf, preset_name, True)
            
            available_presets = "Blue-Red, Cool to Warm, Viridis, Plasma, Magma, Inferno, Rainbow, Grayscale"
            return True, f"Applied color map preset: {preset_name}. Available presets include: {available_presets}"
        except Exception as e:
            self.logger.error(f"Error setting color map: {str(e)}")
            return False, f"Error setting color map: {str(e)}"

    def get_histogram(self, field=None, num_bins=256, data_location="POINTS"):
        """
        Compute and retrieve histogram data for a field in the active data source.
        This function is designed to work with volume sources. By default it uses the
        point data arrays (data_location="POINTS"), but you can specify "CELLS" if your
        volume source stores scalars on cells.

        If no field is provided and the active source contains exactly one available numeric 
        field in the specified data location, that field is automatically used. If multiple 
        arrays exist, the user must specify which field to use.

        Args:
            field (str, optional): The name of the field for which the histogram is computed.
            num_bins (int, optional): Number of histogram bins (default is 10).
            data_location (str, optional): Specify "POINTS" (default) or "CELLS" to indicate the source of the data.
            
        Returns:
            tuple: (success (bool), message (str), histogram_data (list of tuples))
                histogram_data is a list of tuples (bin_center, frequency) representing the computed histogram.

        Note:
            This function uses the Histogram filter from paraview.simple and updates the pipeline.
            Since direct assignment to properties like 'NumberOfBins' is disallowed, the code retrieves
            the proper property (either "NumberOfBins" or "BinCount") via GetProperty() and sets it via SetElement().
        """
        try:
            from paraview.simple import GetActiveSource, Histogram, UpdatePipeline, servermanager
            source = GetActiveSource()
            if not source:
                return False, "Error: No active source. Load data first.", None

            # Obtain the data information from the specified location.
            data_info = source.GetDataInformation()
            data_location = data_location.upper()
            if data_location == "CELLS":
                array_info_obj = data_info.GetCellDataInformation()
            else:
                array_info_obj = data_info.GetPointDataInformation()
            num_arrays = array_info_obj.GetNumberOfArrays()

            # Automatically determine the field if not provided.
            if field is None:
                if num_arrays == 1:
                    field = array_info_obj.GetArrayInformation(0).GetName()
                else:
                    available_arrays = []
                    for i in range(num_arrays):
                        available_arrays.append(array_info_obj.GetArrayInformation(i).GetName())
                    return (
                        False,
                        "Error: Multiple fields available. Please specify a field name. Available arrays: " +
                        ", ".join(available_arrays),
                        None
                    )

            # Create and configure the Histogram filter.
            hist_filter = Histogram(Input=source)
            # Set the input array from the chosen location (POINTS or CELLS).
            hist_filter.SelectInputArray = [data_location, field]

            # Set the number of bins via GetProperty to avoid creating new attributes.
            nbins_prop = hist_filter.GetProperty("NumberOfBins")
            if nbins_prop is None:
                nbins_prop = hist_filter.GetProperty("BinCount")
            if nbins_prop is None:
                return False, "Error: Histogram filter does not have a 'NumberOfBins' or 'BinCount' property.", None
            nbins_prop.SetElement(0, num_bins)

            # Update the pipeline to compute the histogram.
            UpdatePipeline()

            # Fetch the computed histogram (returned as a vtkTable).
            hist_table = servermanager.Fetch(hist_filter)
            if hist_table.GetNumberOfRows() == 0:
                return False, "Histogram computation returned empty data.", None

            # Try to extract histogram data assuming columns named "bin_centers" and "bin_frequencies".
            bin_centers_col = hist_table.GetColumnByName("bin_centers")
            frequencies_col = hist_table.GetColumnByName("bin_frequencies")
            # Fallback: use the first two columns if the expected names do not exist.
            if not bin_centers_col or not frequencies_col:
                bin_centers_col = hist_table.GetColumn(0)
                frequencies_col = hist_table.GetColumn(1)

            histogram_data = []
            num_rows = hist_table.GetNumberOfRows()
            for i in range(num_rows):
                # Retrieve each value from the vtkArray for bin center and frequency.
                bin_center = bin_centers_col.GetValue(i)
                frequency = frequencies_col.GetValue(i)
                histogram_data.append((bin_center, frequency))

            return True, f"Histogram computed for field '{field}' in {data_location} with {num_bins} bins.", histogram_data

        except Exception as e:
            self.logger.error(f"Error computing histogram: {str(e)}")
            return False, f"Error computing histogram: {str(e)}", None


    def set_representation_type(self, rep_type):
        """
        Set the representation type for the active source.
        
        Args:
            rep_type: Representation type (Surface, Wireframe, Points, Volume, etc.)
            
        Returns:
            tuple: (success, message)
        """
        try:
            from paraview.simple import GetActiveSource, GetActiveView, GetDisplayProperties
            source = GetActiveSource()
            if not source:
                return False, "Error: No active source. Load data first."
            
            view = GetActiveView()
            display = GetDisplayProperties(source, view)
            
            display.SetRepresentationType(rep_type)
            
            return True, f"Set representation type to {rep_type}"
        except Exception as e:
            self.logger.error(f"Error setting representation type: {str(e)}")
            return False, f"Error setting representation type: {str(e)}"
    
    def edit_volume_opacity(self, field_name, opacity_points):
        """
        Edit ONLY the opacity transfer function for a given field, ensuring
        we pass only (value, alpha) pairs to ParaView.

        Args:
            field_name (str): The name of the field/array to modify.
            opacity_points (list of tuples): Each tuple must be (value, alpha).
                Example: [(0.0, 0.0), (50.0, 0.3), (100.0, 1.0)]

        Returns:
            tuple: (success: bool, message: str)
        """
        try:
            from paraview.simple import GetOpacityTransferFunction

            if not opacity_points:
                return False, "No opacity points provided."

            # Grab the opacity transfer function for the specified field
            opacity_tf = GetOpacityTransferFunction(field_name)
            if opacity_tf is None:
                return False, f"Could not find an opacity transfer function for field '{field_name}'."

            # Flatten the list of (value, alpha) into the format:
            # [val1, alpha1, midpoint1, sharpness1, val2, alpha2, midpoint2, sharpness2, ...]
            new_opacity_pts = []
            for val, alpha in opacity_points:
                new_opacity_pts.extend([val, alpha, 0.5, 0.0])  # midpoint=0.5, sharpness=0.0

            # Assign them to the piecewise function
            opacity_tf.Points = new_opacity_pts

            return True, f"Opacity transfer function updated for field '{field_name}'."

        except Exception as e:
            self.logger.error(f"Error editing opacity transfer function: {str(e)}")
            return False, f"Error editing opacity transfer function: {str(e)}"


    def set_color_map(self, field_name, color_points):
        """
        Sets the color transfer function for the given field (array) in ParaView.

        Args:
            field_name (str): The name of the field/array (as it appears in ParaView).
            color_points (list of (float, (float, float, float))):
                Each element should be a tuple: (value, (r, g, b))
                where value is the data value, and r, g, b are in [0, 1].

        Returns:
            tuple (success: bool, message: str)
        """
        try:
            from paraview.simple import GetColorTransferFunction

            if not color_points:
                return False, "No color points provided."

            # Retrieve/create the color transfer function for the specified field
            color_tf = GetColorTransferFunction(field_name)
            if color_tf is None:
                return False, f"Could not find or create a color transfer function for '{field_name}'."

            # Flatten the list into [value, R, G, B, value, R, G, B, ...]
            new_rgb_points = []
            for val, rgb in color_points:
                if len(rgb) != 3:
                    return False, f"Invalid RGB tuple for value {val}: {rgb}"
                r, g, b = rgb
                new_rgb_points.extend([val, r, g, b])

            # Update the color transfer function
            color_tf.RGBPoints = new_rgb_points

            # Optionally, you can rescale the transfer function based on min and max values
            # Example:
            # min_val = min([pt[0] for pt in color_points])
            # max_val = max([pt[0] for pt in color_points])
            # color_tf.RescaleTransferFunction(min_val, max_val)

            return True, f"Color transfer function updated for field '{field_name}'."

        except Exception as e:
            msg = f"Error setting color map: {str(e)}"
            return False, msg


    def get_pipeline(self):
        """
        Get the current pipeline structure.
        
        Returns:
            tuple: (success, message)
        """
        try:
            from paraview.simple import GetSources
            sources = GetSources()
            if not sources:
                return True, "Pipeline is empty. No sources found."
            
            response = "Current pipeline:\n"
            for name, source in sources.items():
                response += f"- {name[0]}: {source.__class__.__name__}\n"
            return True, response
        except Exception as e:
            self.logger.error(f"Error getting pipeline: {str(e)}")
            return False, f"Error getting pipeline: {str(e)}"
    
    def get_available_arrays(self):
        """
        Get a list of available arrays in the active source.

        Returns:
            tuple: (success, message)
        """
        try:
            from paraview.simple import GetActiveSource
            source = GetActiveSource()
            if not source:
                return False, "Error: No active source. Load data first."

            # Obtain comprehensive data information from the source.
            data_info = source.GetDataInformation()
            point_info = data_info.GetPointDataInformation()
            cell_info  = data_info.GetCellDataInformation()

            response = "Available arrays:\n\nPoint data arrays:\n"
            if point_info:
                num_point_arrays = point_info.GetNumberOfArrays()
                for i in range(num_point_arrays):
                    # Get the array information for each point array.
                    array_info = point_info.GetArrayInformation(i)
                    array_name = array_info.GetName()  # Use GetName() rather than GetArrayName()
                    components = array_info.GetNumberOfComponents()
                    response += f"- {array_name} ({components} components)\n"
            else:
                response += "No point data arrays found.\n"

            response += "\nCell data arrays:\n"
            if cell_info:
                num_cell_arrays = cell_info.GetNumberOfArrays()
                for i in range(num_cell_arrays):
                    # Get the array information for each cell array.
                    array_info = cell_info.GetArrayInformation(i)
                    array_name = array_info.GetName()
                    components = array_info.GetNumberOfComponents()
                    response += f"- {array_name} ({components} components)\n"
            else:
                response += "No cell data arrays found.\n"

            return True, response
        except Exception as e:
            self.logger.error(f"Error getting available arrays: {str(e)}")
            return False, f"Error getting available arrays: {str(e)}"

    def create_stream_tracer(self, vector_field=None, base_source=None, point_center=None,
                            integration_direction="BOTH", 
                            initial_step_length=0.1,
                            maximum_stream_length=50.0,
                            number_of_streamlines=100,
                            point_radius=1.0,
                            tube_radius=0.1,
                            make_volume_transparent=True):
        """
        Create a stream tracer visualization for a vector volume with tube representation.

        Args:
            vector_field (str, optional): Name of the vector field to trace.
                                        If None, the function automatically selects
                                        the first array with more than one component.
            base_source (optional): The data source (volume) on which to perform stream tracing.
                                If None, uses self.original_source or GetActiveSource().
            point_center (list, optional): Center coordinates [x, y, z] for the seed points.
                                        If None, the center of the volume's bounds is used.
            integration_direction (str): "FORWARD", "BACKWARD", or "BOTH" for integration.
            initial_step_length (float): The initial step size.
            maximum_stream_length (float): Maximum streamline length, beyond which integration terminates.
            number_of_streamlines (int): Number of seed points if a default seed is created.
            point_radius (float): Radius for the Point Cloud seed.
            tube_radius (float): Radius for the tube visualization.
            make_volume_transparent (bool): Whether to make the base volume transparent.

        Returns:
            tuple: (success (bool), message (str), tube filter proxy, tube_name (str))
        """
        try:
            from paraview.simple import (GetActiveSource, GetActiveView, StreamTracer, 
                                        Show, SetActiveSource, FindSource, Tube,
                                        GetDisplayProperties, ColorBy)

            # Determine the base source: use provided, or self.original_source, or the active source.
            if base_source is None:
                base_source = self.original_source or GetActiveSource()
            if not base_source:
                return False, "Error: No active source. Load data first.", None, ""

            # Log the base source name if available.
            base_source_name = None
            if hasattr(base_source, "SMProxy") and hasattr(base_source.SMProxy, "GetXMLName"):
                base_source_name = base_source.SMProxy.GetXMLName()
            else:
                base_source_name = str(base_source)
            self.logger.info(f"Using base source: {base_source_name}")

            # If vector_field is not provided, get the first available multi-component array.
            if vector_field is None:
                # Retrieve the data information and then its point data information.
                data_info = base_source.GetDataInformation()
                point_info = data_info.GetPointDataInformation()
                if point_info:
                    num_arrays = point_info.GetNumberOfArrays()
                    found = False
                    for i in range(num_arrays):
                        array_info = point_info.GetArrayInformation(i)
                        components = array_info.GetNumberOfComponents()
                        if components > 1:
                            vector_field = array_info.GetName()
                            found = True
                            self.logger.info(f"Automatically selected vector field: {vector_field}")
                            break
                    if not found:
                        if num_arrays > 0:
                            vector_field = point_info.GetArrayInformation(0).GetName()
                            self.logger.info(f"No multi-component array found; selected first array: {vector_field}")
                        else:
                            return False, "Error: No arrays found in the base source.", None, ""
                else:
                    return False, "Error: Could not retrieve point data information.", None, ""

            # Determine point center for the seed source
            center = point_center
            if center is None:
                data_info = base_source.GetDataInformation()
                bounds = data_info.GetBounds()  # Format: [xmin, xmax, ymin, ymax, zmin, zmax]
                center = [(bounds[0] + bounds[1]) / 2.0,
                        (bounds[2] + bounds[3]) / 2.0,
                        (bounds[4] + bounds[5]) / 2.0]
                self.logger.info(f"Using auto-calculated center point at {center}")

            # Create the stream tracer filter using Point Cloud seed type
            tracer = StreamTracer(Input=base_source, SeedType='Point Cloud')
            tracer.Vectors = ['POINTS', vector_field]
            tracer.IntegrationDirection = integration_direction
            tracer.InitialStepLength = initial_step_length
            tracer.MaximumStreamlineLength = maximum_stream_length
            
            # Configure the Point Cloud seed
            tracer.SeedType.Center = center
            tracer.SeedType.NumberOfPoints = number_of_streamlines
            tracer.SeedType.Radius = point_radius
            
            # Display the tracer result
            Show(tracer)
            
            # Create tube filter for better visualization
            tube = Tube(Input=tracer)
            tube.Radius = tube_radius
            
            # Show the tube filter
            # Display the tube with proper coloring
            tube_display = Show(tube)
            ColorBy(tube_display, ('POINTS', vector_field))

            # Make the base source transparent if requested
            if make_volume_transparent:
                try:
                    base_display = GetDisplayProperties(base_source)
                    base_display.Opacity = 0.3  # Set opacity to make volume transparent
                except Exception as e:
                    self.logger.warning(f"Could not make volume transparent: {str(e)}")
            
            # Set the active source to the tube filter
            SetActiveSource(tube)
            
            # Get the tube filter name using the helper function
            tube_name = self._get_source_name(tube)

            msg = f"Stream tracer with tubes created for vector field '{vector_field}' using base source '{base_source_name}'."
            return True, msg, tube, tube_name
        except Exception as e:
            self.logger.error(f"Error creating stream tracer: {str(e)}")
            return False, f"Error creating stream tracer: {str(e)}", None, ""

    def get_screenshot(self, width=1280, height=720):
        """
        Capture a screenshot from the current view.

        Args:
            width: Image width in pixels
            height: Image height in pixels

        Returns:
            tuple: (success, message, img_path)
        """
        try:
            import tempfile
            if self.collaboration:
                from paraview.collaboration import processServerEvents
                processServerEvents()
            from paraview import servermanager
            from paraview.simple import SetActiveView, RenderAllViews, SaveScreenshot, ResetCamera, GetActiveViewOrCreate
            
            # Get the active render view from the GUI connection
            pxm = servermanager.ProxyManager()
            gui_view = None
            views = pxm.GetProxiesInGroup("views")
            for (group, name), view_proxy in views.items():
                if view_proxy.GetXMLName() == "RenderView":
                    gui_view = view_proxy
                    break
            
            if not gui_view:
                if self.collaboration:
                    return False, "No render view found. Load data or connect the ParaView GUI first.", ""
                gui_view = GetActiveViewOrCreate("RenderView")
            
            # Set the found GUI view active
            SetActiveView(gui_view)
            RenderAllViews()
            
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                temp_path = tmp.name
            
            SaveScreenshot(temp_path, gui_view, ImageResolution=[int(width), int(height)])
            return True, "Screenshot captured", temp_path
        except Exception as e:
            self.logger.error(f"Error getting screenshot: {str(e)}")
            return False, f"Error getting screenshot: {str(e)}", None
    

    def list_data_files(self, pattern="**/*", limit=200):
        """
        List loadable files below the data directories.

        Args:
            pattern: Glob pattern relative to each data directory
            limit: Maximum number of files to list

        Returns:
            tuple: (success, message, paths)
        """
        if not self.data_roots:
            return False, "Error: No data directories are configured.", []
        try:
            paths = []
            for root in self.data_roots:
                for path in sorted(root.glob(pattern)):
                    if path.is_file():
                        paths.append(str(path))
                        if len(paths) >= limit:
                            return True, f"First {limit} matching files", paths
            return True, f"{len(paths)} matching files", paths
        except Exception as e:
            self.logger.error(f"Error listing data files: {str(e)}")
            return False, f"Error listing data files: {str(e)}", []

    def _time_source(self):
        """The loaded data source, whose time steps drive the animation."""
        from paraview.simple import GetActiveSource
        return self.original_source or GetActiveSource()

    def get_timesteps(self):
        """
        Describe the time steps of the loaded data.

        Returns:
            tuple: (success, message)
        """
        try:
            from paraview.simple import GetAnimationScene
            source = self._time_source()
            if not source:
                return False, "Error: No data loaded."
            values = list(getattr(source, "TimestepValues", None) or [])
            if not values:
                return True, "The loaded data has no time steps."
            current = GetAnimationScene().AnimationTime
            return True, (
                f"{len(values)} time steps from {values[0]:g} to {values[-1]:g} "
                f"(index 0 to {len(values) - 1}); current time {current:g}."
            )
        except Exception as e:
            self.logger.error(f"Error reading time steps: {str(e)}")
            return False, f"Error reading time steps: {str(e)}"

    def set_time(self, index=None, value=None):
        """
        Show the data at a time step.

        Args:
            index: Time step index; negative values count from the end
            value: Time value; the nearest time step is used

        Returns:
            tuple: (success, message)
        """
        try:
            from paraview.simple import GetAnimationScene, Render
            source = self._time_source()
            if not source:
                return False, "Error: No data loaded."
            values = list(getattr(source, "TimestepValues", None) or [])
            if not values:
                return False, "Error: The loaded data has no time steps."
            if index is None and value is None:
                return False, "Error: Give a time step index or a time value."
            if index is not None:
                if not -len(values) <= index < len(values):
                    return False, f"Error: Index {index} is out of range (0 to {len(values) - 1})."
                index = index % len(values)
            else:
                index = min(range(len(values)), key=lambda i: abs(values[i] - value))
            scene = GetAnimationScene()
            scene.UpdateAnimationUsingDataTimeSteps()
            scene.AnimationTime = values[index]
            Render()
            return True, f"Showing time step {index} of {len(values) - 1} (time {values[index]:g})."
        except Exception as e:
            self.logger.error(f"Error setting time: {str(e)}")
            return False, f"Error setting time: {str(e)}"

    def reset_session(self):
        """
        Remove all sources and filters so another analysis can start.

        Returns:
            tuple: (success, message)
        """
        try:
            from paraview.simple import Delete, GetSources, Render
            # Filters must go before their inputs; retry until nothing is left.
            for _ in range(100):
                sources = list(GetSources().values())
                if not sources:
                    break
                for source in sources:
                    try:
                        Delete(source)
                    except Exception:
                        pass
            self.original_source = None
            self._data_folder = ""
            Render()
            return True, "Removed all data and filters."
        except Exception as e:
            self.logger.error(f"Error resetting session: {str(e)}")
            return False, f"Error resetting session: {str(e)}"

    def rotate_camera(self, azimuth=30.0, elevation=0.0):
        """
        Rotate the camera by specified angles.
        
        Args:
            azimuth: Rotation around vertical axis in degrees.
            elevation: Rotation around horizontal axis in degrees.
            
        Returns:
            tuple: (success, message)
        """
        try:
            from paraview.simple import GetActiveView
            view = GetActiveView()
            if not view:
                return False, "Error: No active view."
            
            camera = view.GetActiveCamera()
            camera.Azimuth(azimuth)
            camera.Elevation(elevation)
            return True, f"Rotated camera by azimuth: {azimuth}, elevation: {elevation}"
        except Exception as e:
            self.logger.error(f"Error rotating camera: {str(e)}")
            return False, f"Error rotating camera: {str(e)}"
    
    def reset_camera(self):
        """
        Reset the camera to show all data.
        
        Returns:
            tuple: (success, message)
        """
        try:
            from paraview.simple import GetActiveView, ResetCamera
            view = GetActiveView()
            if not view:
                return False, "Error: No active view."
            ResetCamera(view)
            return True, "Camera reset"
        except Exception as e:
            self.logger.error(f"Error resetting camera: {str(e)}")
            return False, f"Error resetting camera: {str(e)}"
    

    def plot_over_line(self, point1=None, point2=None, resolution=100):
        """
        Create a 'Plot Over Line' filter to sample data along a line between two points.

        Args:
            point1 (list/tuple or None): The [x, y, z] coordinates of the start point. If None, will use data bounds.
            point2 (list/tuple or None): The [x, y, z] coordinates of the end point. If None, will use data bounds.
            resolution (int): Number of sample points along the line (default: 100).

        Returns:
            tuple: (success: bool, message: str, plot_filter)
        """
        try:
            from paraview.simple import GetActiveSource, PlotOverLine, Show, GetActiveView, CreateView, AssignViewToLayout
            source = GetActiveSource()
            if not source:
                return False, "Error: No active source. Load data first.", None

            # Create the PlotOverLine filter
            plot_filter = PlotOverLine(Input=source)
            if point1 is not None:
                plot_filter.Point1 = point1
            if point2 is not None:
                plot_filter.Point2 = point2
            plot_filter.Resolution = resolution

            # Show the result in the active view (usually a line chart view)
            view = GetActiveView()
            Show(plot_filter, view)
            # Create a new 'Line Chart View'
            lineChartView1 = CreateView('XYChartView')

            # show data in view
            plotOverLine1Display_1 = Show(plot_filter, lineChartView1, 'XYChartRepresentation')
            AssignViewToLayout(view=lineChartView1)

            return True, f"Plot over line created from {plot_filter.Point1} to {plot_filter.Point2} with {resolution} points.", plot_filter
        except Exception as e:
            self.logger.error(f"Error creating plot over line: {str(e)}")
            return False, f"Error creating plot over line: {str(e)}", None

    def warp_by_vector(self, vector_field=None, scale_factor=1.0):
        """
        Apply the 'Warp By Vector' filter to the active source.

        Args:
            vector_field (str, optional): The name of the vector field to use for warping. If None, the first available vector field will be used.
            scale_factor (float, optional): The scale factor for the warp (default: 1.0).

        Returns:
            tuple: (success: bool, message: str, warp_filter)
        """
        try:
            from paraview.simple import GetActiveSource, WarpByVector, Show, GetActiveView
            source = GetActiveSource()
            if not source:
                return False, "Error: No active source. Load data first.", None

            # If vector_field is not specified, try to auto-detect a vector field
            if vector_field is None:
                data_info = source.GetDataInformation()
                point_info = data_info.GetPointDataInformation()
                num_arrays = point_info.GetNumberOfArrays()
                found = False
                for i in range(num_arrays):
                    array_info = point_info.GetArrayInformation(i)
                    if array_info.GetNumberOfComponents() > 1:
                        vector_field = array_info.GetName()
                        found = True
                        break
                if not found:
                    return False, "No vector field found in the active source.", None

            # Create the WarpByVector filter
            warp_filter = WarpByVector(Input=source)
            warp_filter.Vectors = ['POINTS', vector_field]
            warp_filter.ScaleFactor = scale_factor

            # Show the result in the active view
            view = GetActiveView()
            Show(warp_filter, view)

            return True, f"Warp by vector applied using field '{vector_field}' with scale factor {scale_factor}.", warp_filter
        except Exception as e:
            self.logger.error(f"Error creating warp by vector: {str(e)}")
            return False, f"Error creating warp by vector: {str(e)}", None
