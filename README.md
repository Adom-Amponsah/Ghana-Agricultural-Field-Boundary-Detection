# 🌾 Ghana Agricultural Field Boundary Detection

This project uses AI-powered methods to detect agricultural field boundaries in Ghana from **.tif geospatial data** (such as maize production raster files).  
It provides both a **model pipeline** and a **Streamlit-based interactive dashboard** for visualization and analysis.

---

## 📌 Features
- Upload a `.tif` geospatial file (e.g., Ghana maize production data for 2024).
- Process the file using multiple delineation methods:
  - Watershed Segmentation
  - K-means Clustering
  - Threshold-based Segmentation
  - Contour Detection
- Visualize:
  - Original production raster
  - Detected field boundaries
  - Overlay maps
- Export results:
  - Boundary maps (PNG)
  - Field statistics (CSV)
  - Summary report (TXT)

---

## ⚙️ Installation & Setup

Clone the repository:
```bash
git clone https://github.com/Adom-Amponsah/Ghana-Agricultural-Field-Boundary-Detection.git
cd Ghana-Agricultural-Field-Boundary-Detection
```

Create and activate a virtual environment (recommended):
```bash
python -m venv venv
# Activate the environment
# On Mac/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

If you don’t have `requirements.txt`, install manually:
```bash
pip install streamlit rasterio matplotlib scikit-image scipy pandas pillow plotly scikit-learn opencv-python
```

---

## ▶️ Running the Application

Start the Streamlit dashboard with:
```bash
streamlit run ghana_field_demo.py
```

This will launch a local server. Open the link shown in your terminal (e.g., `http://localhost:8501`) in your browser.

---

## 📂 Project Structure
```
.
├── ghana_field_demo.py        # Main Streamlit application
├── ghana_field_boundaries.tif # Example output (if available)
├── requirements.txt           # Dependencies
└── README.md                  # Project documentation
```

---

## 📊 Example Workflow
1. Upload your `.tif` maize production raster (e.g., Ghana_Predicted_Maize_Production_2024.tif).  
2. Choose a segmentation method (watershed, kmeans, threshold, contour).  
3. View:
   - Field boundaries
   - Field statistics (count, average size, total area, etc.)  
4. Export results for policymakers, planners, or farmers.  

---

## 🌍 Use Case
This tool helps:
- **Farmers**: understand farm sizes and productivity zones.  
- **Policymakers**: plan agricultural policies and allocate resources.  
- **Researchers**: study field distribution and yield patterns.  

---

## 📦 Requirements

If you want to regenerate the requirements file yourself:
```bash
pip freeze > requirements.txt
```

Here’s the recommended content of `requirements.txt`:
```
streamlit
rasterio
matplotlib
scikit-image
scipy
pandas
pillow
plotly
scikit-learn
opencv-python
```

---

## 👤 Author
Developed by **Adom Amponsah Isaac** 🇬🇭
