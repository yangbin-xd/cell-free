"""Pre-warmed launcher for the Streamlit app.

`streamlit run app.py` imports nothing heavy until the FIRST browser session
connects — that visitor then pays the whole cold start (torch + PyG + the
184 MB CSI + dataset tensors, tens of seconds on a cold page cache). Streamlit
executes app scripts in threads of this same process, so importing the heavy
modules here, before the server starts, moves that cost to service startup:
`sys.modules` is shared and the first session's imports become no-ops.

Usage (replaces `streamlit run app.py` — extra CLI args are passed through):
    python serve.py --server.port=8502 --server.address=127.0.0.1 --server.headless=true
"""
import sys

# Heavy third-party imports first (torch pulls in most of the cost).
import numpy  # noqa: F401
import torch  # noqa: F401
import plotly.graph_objects  # noqa: F401

# Project data modules: loads BS/UE locations, the full CSI array, and the
# train/test tensors once. Must run from the repo root (data/ is relative).
import main       # noqa: F401
import generate   # noqa: F401
import process    # noqa: F401

# GNN model modules (torch_geometric import chain).
import signal_map  # noqa: F401
import interf_map  # noqa: F401
import rate_map    # noqa: F401

from streamlit.web import cli as stcli

if __name__ == "__main__":
    sys.argv = ["streamlit", "run", "app.py"] + sys.argv[1:]
    sys.exit(stcli.main())
