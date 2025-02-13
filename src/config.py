test = True
# ----------------- Sorting parameters -----------------
sort_method = "MST"  # MST, greedy, classic

# ----------------- Tracing parameters -----------------
tracing_method = "interpolation"  # interpolation, classic
interpolation_degree = 3

# ----------------- Shi-Tomasi parameters -----------------
maxcorners = 1000000
minquality = 0.09
mindistance = 10
blocksize = 3

# ----------------- Input directories -----------------
frames_dir = "NRV0719"
masks_dir = "test_masks"
if test:
    frames_output_dir = f"results/{sort_method}-{tracing_method}-test"  # automatically edited
else:
    frames_output_dir = f"results/{sort_method}-{tracing_method}"  # automatically edited


# ----------------- Performance directories -----------------
ground_frames_dir = "drawn-cracks"
gen_frames_dir = f"{frames_output_dir}/pure-lines-out"  # automatically edited
metrics_output_dir = f"{frames_output_dir}/metrics"  # automatically edited
output_json_file = f"{metrics_output_dir}/_metrics.json"  # automatically edited
output_csv_file = f"{metrics_output_dir}/_metrics.csv"  # automatically edited
output_overlay_dir = f"{metrics_output_dir}/overlay"  # automatically edited
