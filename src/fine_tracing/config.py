"""Runtime configuration for fine_tracing.

The original repository stored absolute, machine-specific paths inside the
configuration module (e.g. ``/home/username/...``). For a distributable
package we instead provide **relative defaults** that can be overridden via:

1. CLI arguments (see ``cli.py``) which mutate these globals at runtime.
2. Environment variables (``FINE_TRACING_FRAMES_DIR`` etc.).
3. Direct import & assignment in user scripts.

This is intentionally lightweight; in a future refactor you may want to move
to a dataclass-based immutable settings object.
"""
from __future__ import annotations
import os

# ----------------- Sorting parameters -----------------
sort_method = os.getenv("FINE_TRACING_SORT_METHOD", "classic")  # MST, greedy, classic
greedy_dist_threshold = int(os.getenv("FINE_TRACING_GREEDY_DIST", "10"))

# ----------------- Tracing parameters -----------------
tracing_method = os.getenv("FINE_TRACING_TRACING_METHOD", "classic")  # int, classic
interpolation_degree = int(os.getenv("FINE_TRACING_INTERP_DEG", "3"))  # (range: 1-5)
detection_method = os.getenv("FINE_TRACING_DETECTION", "orb")  # shitomasi, orb

# ----------------- Shi-Tomasi parameters -----------------
maxcorners = int(os.getenv("FINE_TRACING_MAXCORNERS", "1000000"))
minquality = float(os.getenv("FINE_TRACING_MINQUALITY", "0.1"))
mindistance = int(os.getenv("FINE_TRACING_MINDIST", "10"))
blocksize = int(os.getenv("FINE_TRACING_BLOCKSIZE", "3"))
corner_distance_threshold = int(os.getenv("FINE_TRACING_CORNER_DIST", "20"))  # pixel/corner

# ----------------- Input / Output directories (relative defaults) -----------------
frames_dir = os.getenv("FINE_TRACING_FRAMES_DIR", "data/frames")
masks_dir = os.getenv("FINE_TRACING_MASKS_DIR", "data/masks")
input_format = os.getenv("FINE_TRACING_INPUT_FORMAT", "png")  # ppm, png, jpg
output_format = os.getenv("FINE_TRACING_OUTPUT_FORMAT", "png")  # png, jpg, ppm

# Base output directory (can be overridden). The combination of key
# parameters is appended for uniqueness & experiment tracking.
_base_out = os.getenv("FINE_TRACING_OUTPUT_BASE", "results")
frames_output_dir = os.getenv(
	"FINE_TRACING_OUTPUT_DIR",
	f"{_base_out}/{frames_dir}/{sort_method}-{tracing_method}-{minquality}-{corner_distance_threshold}",
)

# ----------------- Crack Length File  -----------------
crack_length_file = f"{frames_output_dir}/crack_length.csv"

# ----------------- Metrics parameters -----------------
metrics_params = [
	"Precision",
	"Recall",
	"F1 Score",
	"IoU",
]

metrics_distance_threshold = int(os.getenv("FINE_TRACING_METRIC_DIST", "5"))

# ----------------- Metrics directories (derived) -----------------
ground_frames_dir = os.getenv("FINE_TRACING_GROUND_DIR", "data/ground-truth")
gen_frames_dir = f"{frames_output_dir}/pure-lines-out"
metrics_output_dir = f"{frames_output_dir}/metrics"
output_json_file = f"{metrics_output_dir}/_metrics.json"
output_csv_file = f"{metrics_output_dir}/_metrics.csv"
output_overlay_dir = f"{metrics_output_dir}/overlay"

# ORB parameters
orb_features = int(os.getenv("FINE_TRACING_ORB_FEATURES", "600"))

