
## PV Crack Detection
### General Usage
---
`python3 -m src.main`


### **Package Files**

* [`config.py`](./config.py): Configuration file containing parameters for the detection process.
* [`greedy.py`](./greedy.py): Implementation of the greedy-heuristic algorithm for crack ordering.
* [`main.py`](./main.py): Main script to run the crack detection process.
* [`metrics.py`](./metrics.py): Script for calculating performance metrics.
* [`MST.py`](./MST.py): Implementation of the Minimum Spanning Tree (MST) algorithm for crack ordering.
* [`shitomasi.py`](./shitomasi.py): Script for processing frames using the Shi-Tomasi algorithm.
---

### **Configuration File Parameters**
Instead of passing options as arguments, you need to edit the configuration file located at `/src/config.py`. The following options are available:

#### **Tracing Parameters**
* `sort_method`: Sorting method (MST, greedy, classic)
* `tracing_method`: Tracing method (interpolation, classic)
* `interpolation_degree`: Degree of interpolation (default is 3)

#### Shi-Tomasi Parameters
* `maxcorners`: Maximum number of corners for Shi-Tomasi detection (default is 1000000)
* `minquality`: Minimum quality level for Shi-Tomasi detection (default is 0.09)
* `mindistance`: Minimum distance between detected corners (default is 10px)
* `blocksize`: Block size for Shi-Tomasi detection (default is 3)

#### Input/Output Directories
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
---