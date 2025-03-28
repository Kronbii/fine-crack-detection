test = False
# ----------------- Sorting parameters -----------------
sort_method = "classic"  # MST, greedy, classic
greedy_dist_threshold = 10

# ----------------- Tracing parameters -----------------
tracing_method = "classic"  # int:interpolation, classic
interpolation_degree = 3  # (range: 1-5)
detection_method = "orb"  # shitomasi, orb

# ----------------- Shi-Tomasi parameters -----------------
maxcorners = 10000000
minquality = 0.1
mindistance = 10
blocksize = 3
corner_distance_threshold = 20  # pixel/corner

# ----------------- Input directories -----------------
frames_dir = "/home/kronbii/YUMA_batches/NRV0406"
masks_dir = "test"
input_format = "png"  # ppm, png, jpg
output_format = "png"  # png, jpg, ppm
if test:
    frames_output_dir = (
        f"results/testing/{frames_dir}/{sort_method}-{tracing_method}-deleteeeee"
    )
else:
    frames_output_dir = f"results/{frames_dir}/{sort_method}-{tracing_method}-{minquality}-{corner_distance_threshold}-edge-orb-concrete"

# ----------------- Crack Length File  -----------------
crack_length_file = f"{frames_output_dir}/crack_length.csv"

# ----------------- Metrics parameters ----------------- #TODO: complete this
metrics_params = [
    "Precision",
    "Recall",
    "F1 Score",
    "IoU",
]
# ----------------- Metrics directories -----------------
metrics_distance_threshold = 5
# automatically edited
ground_frames_dir = "drawn-cracks"
gen_frames_dir = f"{frames_output_dir}/pure-lines-out"
metrics_output_dir = f"{frames_output_dir}/metrics"
output_json_file = f"{metrics_output_dir}/_metrics.json"
output_csv_file = f"{metrics_output_dir}/_metrics.csv"
output_overlay_dir = f"{metrics_output_dir}/overlay"

orb_features = 600
