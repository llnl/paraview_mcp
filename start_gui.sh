#!/bin/sh
# Start a pvserver shared by the ParaView GUI and the MCP server, then the GUI.
#
#   ./start_gui.sh [PORT]
#
# The MCP server connects with --server localhost --port PORT. pvserver runs
# the d3plot reader, so it gets this directory and the venv on its PYTHONPATH.
cd "$(dirname "$0")" || exit 1
PORT=${1:-11111}
SITE_PACKAGES=$(echo "$PWD"/.venv/lib/python3*/site-packages)
export PYTHONPATH="$PWD:$SITE_PACKAGES${PYTHONPATH:+:$PYTHONPATH}"

pvserver --multi-clients --server-port="$PORT" &
PVSERVER=$!
trap 'kill $PVSERVER 2>/dev/null' EXIT INT TERM
sleep 2
paraview --server-url="cs://localhost:$PORT"
