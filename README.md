# ⛰️ BhuAlert

<p align="center">
  <img src="http://img.lrp.pp.ua/u/36n9pa.png" width="110" alt="BhuAlert Logo"/>
</p>

<p align="center">
  <strong>Real-time landslide risk assessment and hazard-avoiding route advisory system.</strong><br>
  <em>Smart India Hackathon (SIH 2026) Prototype</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/XGBoost-118833?style=for-the-badge&logo=xgboost&logoColor=white" alt="XGBoost"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn"/>
  <img src="https://img.shields.io/badge/Leaflet%20%2F%20Folium-77B829?style=for-the-badge&logo=leaflet&logoColor=white" alt="Folium"/>
  <img src="https://img.shields.io/badge/OSRM%20Routing-7EBC6F?style=for-the-badge&logo=openstreetmap&logoColor=white" alt="OSRM"/>
  <img src="https://img.shields.io/badge/SIH-2026-blueviolet?style=for-the-badge" alt="SIH 2026"/>
</p>

---

## Overview

BhuAlert is a web application that assesses landslide risk for any coordinate in real time and suggests safe driving routes that steer clear of active landslide zones.

Instead of relying on heavy, pre-downloaded GIS files, the app dynamically pulls live elevation and meteorological data from open APIs, passes the processed features to a trained XGBoost model, and renders the results on interactive Folium maps.

---

## How It Works

1. **User Input:** Enter GPS coordinates (Latitude & Longitude) for an area of interest.
2. **Live Data Fetching:**
   - **Slope:** Queries 3 nearby points from the Open-Meteo Elevation API and calculates the terrain slope angle.
   - **Weather:** Queries the Open-Meteo Forecast API for live rainfall (mm/hr), rain category, temperature, and soil moisture (3–9 cm depth).
3. **Risk Prediction:** A scikit-learn pipeline scales numerical inputs, one-hot encodes the rain condition, and feeds them into an XGBoost classifier to output a Risk Score (0–100%).
4. **Interactive Visualization:**
   - Displays a satellite view of the location with a 1.5 km risk buffer circle.
   - If risk is High (≥ 60%), emergency contacts (112, 108, 101) are displayed.
5. **Safe Route Advisory:**
   - The user inputs their current location and target destination.
   - The app checks if the direct route crosses into the landslide danger area.
   - If blocked, it computes an alternate road route around the perimeter using OSRM and renders it in Google Maps navigation style.

---

## Key Features

- **Dynamic Slope Engine:** Computes ground slope in degrees on the fly using trigonometric gradients—no local DEM rasters required.
- **Live Environmental Data:** Real-time rainfall rate, rain classification, temperature, and root-zone soil moisture.
- **Fast ML Inference:** Model training and pipeline setup are cached using `@st.cache_resource`, ensuring instantaneous predictions.
- **Dual Map Views:**
  - **Satellite View:** Esri world imagery centered on the evaluated hazard point.
  - **Navigation View:** Google Maps roadmap layer showing turn-by-turn route lines.
- **Landslide-Avoiding Routing:** Identifies road clearance using the Haversine formula and generates drivable bypass detours around active risk perimeters.

---

## Tech Stack

| Technology | Badge / Logo | Role |
|---|:---:|---|
| **Python** | ![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white) | Core programming language |
| **Streamlit** | ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white) | Web application framework and dashboard UI |
| **XGBoost** | ![XGBoost](https://img.shields.io/badge/XGBoost-118833?style=flat-square&logo=xgboost&logoColor=white) | Gradient-boosted decision trees for hazard classification |
| **Scikit-Learn** | ![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white) | Data preprocessing pipeline (`StandardScaler`, `OneHotEncoder`) |
| **Pandas** | ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white) | Data manipulation and tabular feature matrix handling |
| **NumPy** | ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white) | Numerical operations and array math |
| **Folium** | ![Folium](https://img.shields.io/badge/Folium-77B829?style=flat-square&logo=leaflet&logoColor=white) | Leaflet.js-based interactive map visualization |
| **OSRM** | ![OSRM](https://img.shields.io/badge/OSRM-7EBC6F?style=flat-square&logo=openstreetmap&logoColor=white) | Turn-by-turn road network routing engine |
| **Open-Meteo** | ![Open-Meteo](https://img.shields.io/badge/Open--Meteo-007ACC?style=flat-square&logo=cloud&logoColor=white) | Free live elevation, forecast rain, and soil moisture APIs |
| **Google Maps** | ![Google Maps](https://img.shields.io/badge/Google%20Maps-4285F4?style=flat-square&logo=googlemaps&logoColor=white) | Roadmap tile imagery for clean driving route display |

---

## Project Structure

```text
SIH2026/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
│
├── Datasets/
│   ├── Northeast_India_ML_Simulated_Triggers.csv   # Training dataset
│   ├── Northeast_India_Landslide_Data.csv          # Historical regional records
│   ├── datafile.csv                                # Reference dataset
│   └── landslide_model.pkl                         # Baseline model pickle
│
└── assets/
    └── Icons/                                      # UI logos & graphics
```

---

## Installation & Setup

### 1. Prerequisites
- Python 3.10 or higher
- Active internet connection (for live weather & routing APIs)

### 2. Navigate to Project Directory
```bash
cd C:\Users\subho\Desktop\SIH2026
```

### 3. Create & Activate Virtual Environment
**Windows (PowerShell):**
```powershell
python -m venv SIH.venv
.\SIH.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
python -m venv SIH.venv
.\SIH.venv\Scripts\activate.bat
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Run the Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## Methodologies & Formulas

### 1. Terrain Slope Calculation
Slope is estimated from 3 elevation points spaced by offset $\Delta = 0.001^\circ$ (~111 m):
- Center: $(lat, lon)$
- North: $(lat + \Delta, lon)$
- East: $(lat, lon + \Delta)$

$$\Delta y = \Delta \times 111320 \text{ meters}$$
$$\Delta x = \Delta \times 111320 \times \cos(lat) \text{ meters}$$
$$\text{Rise}_N = \frac{\text{Elev}_N - \text{Elev}_C}{\Delta y}, \quad \text{Rise}_E = \frac{\text{Elev}_E - \text{Elev}_C}{\Delta x}$$
$$\text{Slope} (\text{degrees}) = \arctan\left(\sqrt{\text{Rise}_N^2 + \text{Rise}_E^2}\right) \times \frac{180}{\pi}$$

---

### 2. Hazard Distance (Haversine Formula)
To check whether any coordinate on a road route $(p_{\text{lat}}, p_{\text{lon}})$ enters the hazard perimeter $(h_{\text{lat}}, h_{\text{lon}})$:

$$a = \sin^2\left(\frac{\Delta lat}{2}\right) + \cos(p_{\text{lat}}) \cdot \cos(h_{\text{lat}}) \cdot \sin^2\left(\frac{\Delta lon}{2}\right)$$
$$d = 2 R \cdot \text{atan2}\left(\sqrt{a}, \sqrt{1 - a}\right) \quad (R = 6371 \text{ km})$$

If $\min(d) < \text{Safety Buffer}$ (default 1.5 km), the route is flagged as compromised.

---

### 3. Detour Routing Logic
1. Computes the direct driving route from Origin to Destination via OSRM.
2. Checks if any coordinate along the direct route falls within the hazard radius.
3. If it does, candidate waypoints are projected perpendicularly and radially outside the danger perimeter ($1.6\times$ and $2.4\times$ buffer radius).
4. Evaluates driving routes through candidate waypoints via OSRM.
5. Selects the shortest drivable route that remains completely outside the danger zone.

---

## Route Map Legend

| Indicator | Map Symbol | Description |
|---|---|---|
| **Origin** | Blue dot with halo | User's starting location |
| **Destination** | Red pin | Target destination |
| **Hazard Zone** | Shaded red circle | 1.5 km active landslide danger perimeter |
| **Safe Route** | Solid blue line | Recommended bypass route |
| **Blocked Path** | Dashed red line | Hazardous direct segment to avoid |

---

## SIH 2026
- **Track:** Disaster Management / Intelligent Transportation
- **Focus:** Early warning and road corridor safety for hilly and landslide-prone terrains.
