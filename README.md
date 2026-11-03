# Paraview_MCP

ParaView-MCP is an autonomous agent that integrates multimodal large language models with ParaView through the Model Context Protocol, enabling users to create and manipulate scientific visualizations using natural language and visual inputs instead of complex commands or GUI operations. The system features visual feedback capabilities that allow it to observe the viewport and iteratively refine visualizations, making advanced visualization accessible to non-experts while augmenting expert workflows with intelligent automation.

## Disclaimer 
The current implementation of the connection between the MCP server and ParaView, in both the main and dev branches, relies on synchronization between pvserver and the ParaView client. Because this feature has been deprecated in most recent ParaView versions, the ParaView application view may not display content from the pvserver instance correctly, and overall stability issues may occur.

## Video Demo

Click the image below to watch the video:

[![Video Title](https://img.youtube.com/vi/GvcBnAcIXp4/maxresdefault.jpg)](https://youtu.be/GvcBnAcIXp4)


## Installation

```shell
git clone https://github.com/LLNL/paraview_mcp.git
cd paraview_mcp

conda create -n paraview_mcp python=3.10
conda install conda-forge::paraview
conda install mcp[cli] httpx
```

## Setup for LLM

To set up integration with claude desktop, add the following to claude_desktop_config.json

```json
    "mcpServers": {
      "ParaView": {
        "command": "/path/to/python",
        "args": [
        "/path/to/paraview_mcp/paraview_mcp_server.py"
        ]
      }
    }
```

## running 

### 1. Start paraview server

```shell
python pvserver --multi-clients
```

### 2. Connect to paraview server from paraview GUI (file -> connect)

### 3. Start claude desktop app 

### Claude Code

With ParaView from apt (Ubuntu 24.04 ships 5.11 for the system Python):

```shell
sudo apt install paraview python3-paraview
uv venv --python /usr/bin/python3.12 --system-site-packages .venv
uv pip install --python .venv/bin/python -r requirements.txt
claude mcp add paraview -s user -- "$PWD/.venv/bin/python" "$PWD/paraview_mcp_server.py" --server localhost --port 11111
```

Start `pvserver --multi-clients --server-port=11111` before Claude Code, which
connects once at startup.

## Headless server

`--server builtin` renders in the server process without pvserver or a GUI, and
`--transport streamable-http` serves MCP over HTTP at `http://<host>:<port>/mcp`:

```shell
python paraview_mcp_server.py --server builtin --transport streamable-http \
    --http-host 0.0.0.0 --http-port 8765 --data-root /data --output-dir /output
```

| Option | Meaning |
|---|---|
| `--data-root DIR` | Only load data below DIR (repeatable); relative paths resolve against the first root |
| `--output-dir DIR` | Write files such as STL exports to DIR instead of next to the data |
| `--probe-timeout SECONDS` | Open each file in a subprocess first and reject it when that takes longer (default 120, 0 disables). Truncated files can otherwise block the server forever, e.g. LS-DYNA d3plot files without their state files |
| `--allowed-host HOST:PORT` | Accepted Host headers (repeatable, `*` as port wildcard). Without it, a server listening on a non-loopback address accepts any host |

Additional tools: `list_data_files`, `get_timesteps`, `set_time` (time steps of
transient data such as crash simulations), `render_animation` (the view over
time as image frames), `list_time_series` and `get_time_series` (LS-DYNA
binout history output of one or more runs, as data series), `reset_session`,
and an explicit color range in `color_by` (`range_min`, `range_max`).

Data series are returned as an embedded resource of type
`application/vnd.onion.series+json`:
`{"title", "x": {"label", "values"}, "series": [{"name", "values"}]}`, plus a
text summary for the model.

### FEMZIP-compressed LS-DYNA results

`load_data` reads FEMZIP-compressed d3plot files (which ParaView's LS-DYNA
reader cannot open) through `femzip_reader.py`, using the SIDACT FEMZIP library
bundled with lasso-python 1.5.2. They provide the deformed mesh per time step
without eroded elements, the point array `Deflection` and the cell arrays
`EffPlastStrn` (shells) and `PartID`. Install it next to the requirements:

```shell
uv pip install --python .venv/bin/python --no-deps "lasso-python==1.5.2"
uv pip install --python .venv/bin/python rich psutil h5py "pandas<2.3" "numpy==1.26.4"
```

Pin numpy to the version of the system ParaView (`python3 -c "import numpy; print(numpy.__version__)"`).
Without lasso-python, FEMZIP files are rejected after the probe timeout.

The Docker image runs the headless server on Mesa inside Xvfb:

```shell
docker build -t paraview-mcp .
docker run --rm -p 8765:8765 -v /path/to/data:/data:ro paraview-mcp
```

***

## Citing Paraview_MCP

S. Liu, H. Miao, and P.-T. Bremer, “Paraview-MCP: Autonomous Visualization Agents with Direct Tool Use,” in Proc. IEEE VIS 2025 Short Papers, 2025, pp. 00

```bibtex
@inproceedings{liu2025paraview,
  title={Paraview-MCP: Autonomous Visualization Agents with Direct Tool Use},
  author={Liu, S. and Miao, H. and Bremer, P.-T.},
  booktitle={Proc. IEEE VIS 2025 Short Papers},
  pages={00},
  year={2025},
  organization={IEEE}
}
```

## Authors 
Paraview_MCP was created by Shusen Liu (liu42@llnl.gov) and Haichao Miao (miao1@llnl.gov)

## License
Paraview_MCP is distributed under the terms of the BSD-3 license.

LLNL-CODE-2007260
