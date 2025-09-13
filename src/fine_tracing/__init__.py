"""Fine Tracing - crack tracing & metrics toolkit.

Public submodules:
    config
    preprocessing
    edge
    process
    metrics
    utils

CLI entrypoint provided via fine-tracing command (see cli module).
"""
from . import config, preprocessing, edge, process_corners as process, metrics, utils  # re-export for convenience

__all__ = [
    "config",
    "preprocessing",
    "edge",
    "process",
    "metrics",
    "utils",
]
