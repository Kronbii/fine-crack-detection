sort_method = "greedy"  # MST, greedy, classic
tracing_method = "interpolation"  # interpolation, classic
interpolation_degree = 3
maxcorners = 1000000
minquality = 0.09
mindistance = 10
blocksize = 3

frames_dir = "NRV0719"
masks_dir = "test_masks"
frames_output_dir = f"delete/model2.0/{sort_method}-{tracing_method}"

# Directories
ground_frames_dir = "drawn-cracks"
gen_frames_dir = f"{frames_output_dir}/pure-lines-out"
metrics_output_dir = f"{frames_output_dir}/metrics"
output_json_file = f"{metrics_output_dir}/_metrics.json"
output_csv_file = f"{metrics_output_dir}/_metrics.csv"
