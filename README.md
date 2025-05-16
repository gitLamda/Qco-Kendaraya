# QCO කේන්දරය

**Quick Changeover Prediction Dashboard**

A Streamlit application developed during my internship at MAS Linea Aqua to predict and analyze garment changeover efficiency using multiple machine learning models.

---

## Overview

QCO කේන්දරය integrates three predictive models to provide a comprehensive assessment of production changeover success:

1. **Historia (Classification Model)**  
   - Weight: 30%  
   - Based on historical performance data.  

2. **Critical Path (Regression Model)**  
   - Weight: 60%  
   - Focuses on process and operational factors.  

3. **Talento (Classification Model)**  
   - Weight: 10%  
   - Incorporates team skill and resource factors.  

The application aggregates model outputs into a final score, presents visual insights, and offers data-driven recommendations.

---

## Key Features

- **Interactive Dashboards**  
  Gauge charts, bar charts, and radar plots for model outputs and comparisons.  

- **Custom Styling**  
  Professional UI with a clean, modern design.  

- **Data Upload and Selection**  
  Supports Excel file uploads (`.xlsx`, `.xls`), with dynamic Module and Style selection.  

- **Model Configuration**  
  Toggle individual models on or off and adjust weights.  

- **Detailed Reporting**  
  Tabular breakdown of raw scores, weighted impacts, and final recommendations.

---

## Technology Stack

- **Streamlit** – Application framework  
- **Pandas & NumPy** – Data processing  
- **Plotly** – Data visualization  
- **Scikit‑learn** – Machine learning models  
- **Pickle** – Model serialization  

---

## Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/your-username/qco-kendara.git
   cd qco-kendara

2. **Create and activate a virtual environment**
   python3 -m venv .venv
   source .venv/bin/activate    # macOS/Linux
   .venv\Scripts\activate       # Windows

3. **Install dependencies**
   pip install -r requirements.txt

4. **Place model files in the project root**
    - new_rf_qco.pkl
    - lr_qco.pkl
    - qco_predictor.pkl

5. **Run the app**
   streamlit run app.py

**Contributing**
Contributions and suggestions are welcome. Please open an issue or submit a pull request with improvements or bug fixes.
     

