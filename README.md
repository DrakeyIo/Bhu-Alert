# ⛰️ BhuAlert

> Real-time landslide risk assessment and hazard-avoiding route advisory system built with Streamlit, XGBoost, and OSRM. Developed for SIH 2026.

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

| Component | Tool / Library | Role |
|---|---|---|
| **Web Framework** | Streamlit | UI, inputs, metric cards, and layout |
| **Machine Learning** | XGBoost, Scikit-Learn | Classification pipeline, scaling, and inference |
| **Data Handling** | Pandas, NumPy | Data manipulation and feature formatting |
| **Mapping** | Folium, Streamlit-Folium | Interactive map rendering |
| **Routing** | OSRM (Open Source Routing Machine) | Driving directions and road geometry |
| **Data APIs** | Open-Meteo (Elevation & Forecast) | Live elevation, rain, and soil moisture |

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
