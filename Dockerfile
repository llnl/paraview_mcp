# Headless ParaView MCP server, reachable over streamable HTTP.
#
#   docker build -t paraview-mcp .
#   docker run --rm -p 8765:8765 -v /path/to/data:/data:ro paraview-mcp
#
# The MCP endpoint is http://<host>:8765/mcp. Rendering runs on Mesa inside
# Xvfb, so no GPU or display is needed.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PARAVIEW_MCP_LOG_DIR=/tmp/paraview_logs

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        paraview python3-paraview python3-venv xvfb xauth libgl1-mesa-dri tini \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# The venv sees the apt ParaView module through the system site packages.
# lasso-python 1.5.2 bundles SIDACT's FEMZIP library for reading compressed
# d3plot files. Its declared dependencies cannot be installed (it requires the
# removed "sklearn" package), so the ones the reader needs are listed here.
RUN python3 -m venv --system-site-packages /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt \
    && /opt/venv/bin/pip install --no-cache-dir --no-deps "lasso-python==1.5.2" \
    && /opt/venv/bin/pip install --no-cache-dir rich psutil h5py "pandas<2.3" "numpy==1.26.4"
COPY paraview_manager.py paraview_mcp_server.py d3plot_reader.py time_series.py ./

RUN mkdir -p /data /output && chown ubuntu:ubuntu /output
USER ubuntu

EXPOSE 8765
# xvfb-run hangs as PID 1 because it waits for a signal from Xvfb; tini reaps
# and forwards signals instead.
ENTRYPOINT ["tini", "--", "xvfb-run", "-a", "-s", "-screen 0 1920x1080x24", \
    "/opt/venv/bin/python", "paraview_mcp_server.py", \
    "--server", "builtin", "--transport", "streamable-http", \
    "--http-host", "0.0.0.0", "--http-port", "8765"]
CMD ["--data-root", "/data", "--output-dir", "/output"]
