# **Fine Crack Detection: High-Precision Micro Crack Analysis with CV & AI**  
**_A Deep Learning & Computer Vision Approach for Detecting Ultra-Fine Cracks_**  

[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)  
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)  
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-red.svg)](https://opencv.org/)  
[![SciPy](https://img.shields.io/badge/Scipy-Supported-orange.svg)](https://scipy.org/)  

## **🔍 Project Overview**
Fine Crack Detection is an advanced **computer vision and deep learning framework** designed to detect **ultra-fine cracks** that traditional vision methods fail to identify. This system processes **high-resolution images**, extracts meaningful crack contours, and provides **quantitative performance analysis** by comparing results against ground truth data.  

This tool is ideal for **civil engineers, materials scientists, and AI researchers** working in defect detection, structural health monitoring, and quality assurance.

---

## **🚀 Features**
✅ **High-Precision Crack Detection** – Detects cracks that are almost imperceptible to classical CV methods.  
✅ **Deep Learning & AI Integration** – Supports models trained for **fine crack segmentation and classification**.  
✅ **Comprehensive Performance Metrics** – Computes **Precision, Recall, F1-score, IoU, Mean Distance, RMS Error**.  
✅ **Efficient Contour Interpolation** – Uses **SciPy’s cKDTree and OpenCV contours** for smooth crack representation.  
✅ **Flexible Input Support** – Works with **various image formats** and adapts to different resolutions.  
✅ **Customizable Output** – Generates **CSV reports, JSON summaries, and overlaid visualizations** for analysis.  

---

## **🖼️ Sample Output**
| Ground Truth (Green) | Predicted Crack (Red) | Overlay (Comparison) |
|----------------------|----------------------|----------------------|
| ![Ground Truth](docs/ground_truth.png) | ![Generated](docs/generated.png) | ![Overlay](docs/overlay.png) |

---

## **📂 Project Structure**
```
fine-crack-detection/
│── src/                        # Core source code
│   │── main.py                 # Entry point for crack detection
│   │── performance.py          # Evaluates detection accuracy
│   │── visualization.py        # Generates overlays & plots
│   │── config.py               # Configurations & paths
│   │── utils/                  # Helper functions
│       │── image_processing.py # Image processing utilities
│       │── metrics.py          # Performance metrics calculations
│── data/                       # Input images & ground truth
│── results/                    # Output metrics & overlayed images
│── requirements.txt            # Dependencies
│── README.md                   # Documentation
│── LICENSE                     # License file
```

---

## **⚙️ Installation**
### **1️⃣ Set Up a Virtual Environment**
```bash
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate      # Windows
```

### **2️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

---

## **📊 How It Works**
### **1️⃣ Run the Crack Detection Pipeline**
```bash
python src/main.py
```
- Loads images from `data/`
- Applies **preprocessing, contour detection, and AI model inference**
- Saves detected cracks and overlays in `results/`

### **2️⃣ Evaluate Model Performance**
```bash
python src/performance.py
```
- Computes **Precision, Recall, IoU, F1-score**
- Outputs **CSV & JSON reports** for accuracy comparison

### **3️⃣ Visualize Detection Results**
```bash
python src/visualization.py
```
- Generates **overlay images** with **ground truth (green) vs. predicted cracks (red)**
- Saves visual outputs in `results/`

---

## **📈 Performance Metrics**
This framework evaluates crack detection accuracy using **seven key metrics**:
| Metric          | Description |
|----------------|------------|
| **Precision**  | % of detected cracks that are actually cracks |
| **Recall**     | % of real cracks that were detected |
| **F1-score**   | Harmonic mean of precision & recall |
| **IoU (Jaccard Index)** | Overlap between predicted & ground truth masks |
| **Mean Distance** | Avg. pixel-wise error between real & detected cracks |
| **Max Distance**  | Max pixel-wise deviation |
| **RMS Error**   | Root mean square error of detected crack locations |

**Example CSV Output (`results/crack_metrics.csv`):**
```
Frame,Crack,Precision,Recall,F1-score,IoU,Mean Distance,Max Distance,RMS Error
1897,Crack 1,0.87,0.92,0.89,0.76,2.3,5.0,1.5
1835,Crack 2,0.93,0.95,0.94,0.82,1.8,4.0,1.2
```

---

## **🛠️ Technologies Used**
✅ **Python 3.8+** – Core programming language  
✅ **OpenCV** – Image processing & contour detection  
✅ **NumPy & SciPy** – Fast numerical operations & interpolation  
✅ **Matplotlib** – Visualization of crack overlays  
✅ **Pandas** – CSV and JSON report generation  
✅ **cKDTree (SciPy)** – Fast nearest-neighbor search for crack matching  

---

## **📬 Contact & Contributions**
💡 Have suggestions or improvements? Feel free to **fork the repo & submit a PR**!  

📧 **Contact me:** kronbii@github.com  
🔗 **LinkedIn:** [linkedin.com/in/kronbii](#)  

---

## **📜 License**
This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

## **🌟 If You Find This Useful, Give It a Star!**
If this repository helped you in your work or research, consider giving it a ⭐ on GitHub! Your support helps improve this project! 🚀
