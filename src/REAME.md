# Automated Model Conversion

## Model Conversion
### General Usage
`python3 -m src.main`

### Configuration File
Instead of passing options as arguments, you need to edit the configuration file located at `/src/config.py`. The following options are available:

* `sort_method`: Sorting method (MST, greedy, classic)
* `tracing_method`: Tracing method (interpolation, classic)
* `interpolation_degree`: Degree of interpolation (default is 3)
* `maxcorners`: Maximum number of corners for Shi-Tomasi detection (default is 1000000)
* `minquality`: Minimum quality level for Shi-Tomasi detection (default is 0.09)
* `mindistance`: Minimum distance between detected corners (default is 10px)
* `blocksize`: Block size for Shi-Tomasi detection (default is 3)
* `frames_dir`: Directory for input framess
* `masks_dir`: Directory for input masks
* `ground_frames_dir`: Directory for ground truth frames

After setting the previous parameters, the following parameters will be automatically set
* `frames_output_dir`: Directory for output frames (automatically edited)
* `gen_frames_dir`: Directory for generated frames (automatically edited)
* `metrics_output_dir`: Directory for performance metrics (automatically edited)
* `output_json_file`: Path to output JSON file for metrics (automatically edited)
* `output_csv_file`: Path to output CSV file for metrics (automatically edited)
* `output_overlay_dir`: Directory for overlay output (automatically edited)

Make sure to update the configuration file with the appropriate values before running the script.