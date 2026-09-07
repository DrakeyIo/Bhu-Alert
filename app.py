# =============================================================================
# app.py  -  Landslide Risk Assessment App
# =============================================================================
#
# OVERALL FLOW (read this first):
#
#   1. IMPORTS      - bring in every library we need
#   2. TRAIN MODEL  - load CSV, create labels, train XGBoost, cache it
#   3. FUNCTIONS    - small helper functions used later
#   4. SIDEBAR      - the left panel: coordinate inputs + button
#   5. MAIN PAGE    - the right area: banner + metrics + map + table
#
# Streamlit re-runs this whole file top-to-bottom on every button click.
# @st.cache_resource stops the model from re-training on every click.
# =============================================================================


# =============================================================================
# PART 1 - IMPORTS
# Each line brings in a library (a collection of pre-built tools).
# =============================================================================

import sys           # used to detect if we are running on Windows
import asyncio       # async event loop - needed for a Windows-specific fix
import math          # math functions like cos(), radians(), atan(), sqrt()
import pathlib       # lets us build file paths that work on any OS

import numpy as np   # numerical arrays and random number generation
import pandas as pd  # tables (DataFrames) - the model expects its input as a table

import streamlit as st   # the web app framework - creates buttons, text, maps etc.
import folium            # creates interactive maps
import requests          # makes HTTP calls to web APIs (Open-Meteo)
from streamlit_folium import st_folium  # embeds a Folium map inside Streamlit

# scikit-learn: the machine learning toolkit
from sklearn.pipeline       import Pipeline          # chains preprocessing + model together
from sklearn.compose        import ColumnTransformer # applies different transforms to different columns
from sklearn.preprocessing  import StandardScaler    # scales numbers to a common range
from sklearn.preprocessing  import OneHotEncoder     # converts text categories to numbers
from sklearn.model_selection import train_test_split # splits data into train and test sets

from xgboost import XGBClassifier  # the actual ML model (gradient boosted trees)


# =============================================================================
# PART 2 - BASIC SETUP
# =============================================================================

# Windows has a bug with asyncio - this one line fixes it
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Configure the browser tab: title, icon, layout
st.set_page_config(
    page_title="Landslide Risk Monitor",
    page_icon="http://img.lrp.pp.ua/u/caFRTN.png",   # mountain emoji (unicode so it doesn't corrupt)
    layout="wide"                  # use the full browser width
)

# Build the path to our CSV file.
# pathlib.Path(__file__).parent = the folder containing app.py
# / "Datasets" / "..." = adds subfolders to the path
DATA_PATH = pathlib.Path(__file__).parent / "Datasets" / "Northeast_India_ML_Simulated_Triggers.csv"


# =============================================================================
# PART 3 - TRAIN THE MODEL
# =============================================================================
# @st.cache_resource is a decorator.
# It wraps train_model() so that:
#   - First call: actually runs the function and stores the result
#   - Every later call: skips the function, returns the stored result instantly
# Without this, the model would re-train on every button click (very slow).

@st.cache_resource
def train_model():

    # ---- Load the CSV into a DataFrame (a table) ----
    df = pd.read_csv(DATA_PATH)

    # ---- Fix soil moisture units ----
    # The CSV stores soil moisture as m3/m3 (e.g. 0.79)
    # Multiplying by 100 converts it to percentage (79%)
    # We need % because that's what the live weather API also returns
    df["soil_moisture"] = df["soil_moisture"] * 100

    # ---- Create the target label (what we want the model to predict) ----
    # 1 = this location is at risk of a landslide
    # 0 = this location is safe
    # We don't have real labels, so we create them using a risk scoring rule.

    def calculate_risk(row):
        score = 0

        # Steep slopes are dangerous (but NER data all has slope > 25, so we use 35)
        if row["inclination"] > 35:
            score += 40
        if row["inclination"] > 50:
            score += 20   # extra points for extremely steep ground

        # Very wet soil makes slopes unstable
        if row["soil_moisture"] > 75:
            score += 20

        # Heavy rain adds water to the soil, increasing risk
        rain_points = {
            "downpour":        40,
            "continuous_rain": 20,
            "rain":            10,
        }
        score += rain_points.get(row["rain_report (trigger)"], 0)

        # If total score is 70 or above, label it as "at risk"
        return 1 if score >= 70 else 0

    # Apply calculate_risk to every row in the table
    # axis=1 means "go row by row" (axis=0 would be column by column)
    df["Target"] = df.apply(calculate_risk, axis=1)

    # ---- Select the 4 input features and the target column ----
    X = df[["inclination", "soil_moisture", "temperature", "rain_report (trigger)"]]
    y = df["Target"]

    # ---- Split data: 80% for training, 20% for testing ----
    # random_state=42 makes the split reproducible (same split every time)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ---- Set up the preprocessor ----
    # The model can only work with numbers, so we need to:
    # (a) Scale the 3 number columns so they're all on a similar scale
    # (b) Convert the text column ("rain", "downpour") into numbers
    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), ["inclination", "soil_moisture", "temperature"]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["rain_report (trigger)"]),
    ])
    # handle_unknown="ignore" = if a new rain category appears at prediction time,
    # just treat it as zeros instead of crashing

    # ---- Build the Pipeline ----
    # Pipeline chains steps in order:
    #   Step 1: preprocessor transforms the data
    #   Step 2: XGBClassifier receives the transformed data and learns from it
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(
            n_estimators=100,    # build 100 decision trees
            max_depth=5,         # each tree can ask max 5 yes/no questions
            learning_rate=0.1,   # how much each new tree corrects the previous ones
            eval_metric="logloss",  # how to measure error during training
            random_state=42
        ))
    ])

    # ---- Train the model ----
    # .fit() = show the model all the training examples so it can learn patterns
    pipeline.fit(X_train, y_train)

    # ---- Evaluate on the test set ----
    # .score() = what fraction of test examples did the model get right?
    accuracy = pipeline.score(X_test, y_test)

    # Return both the trained model and the accuracy percentage
    return pipeline, round(accuracy * 100, 2)


# This line actually calls train_model().
# model = the trained pipeline (preprocessor + XGBoost)
# accuracy = e.g. 91.55 (meaning 91.55% correct on test data)
model, accuracy = train_model()


# ---- Sanity check: do the LIVE weather categories match what the model TRAINED on? ----
# get_weather() below produces one of these 5 category strings from live rain (mm/hr).
# If the CSV's "rain_report (trigger)" column used different text for the same idea
# (e.g. "no_rain" instead of "clear", or "heavy_rain" instead of "downpour"), then
# OneHotEncoder(handle_unknown="ignore") will silently encode every live reading in
# that category as all-zeros - i.e. rain has ZERO effect on the prediction, with no
# error or warning. That alone can make genuinely rainy, at-risk locations score low.
# This check just surfaces the mismatch loudly instead of letting it fail silently.
_LIVE_RAIN_CATEGORIES = {"clear", "light_rain", "rain", "continuous_rain", "downpour"}
try:
    _TRAINED_RAIN_CATEGORIES = set(
        model.named_steps["preprocessor"].named_transformers_["cat"].categories_[0]
    )
except Exception:
    _TRAINED_RAIN_CATEGORIES = _LIVE_RAIN_CATEGORIES  # if this ever fails, skip the check
_UNSEEN_RAIN_CATEGORIES = _LIVE_RAIN_CATEGORIES - _TRAINED_RAIN_CATEGORIES


# =============================================================================
# PART 4 - HELPER FUNCTIONS
# Small reusable functions that are called later in the app.
# =============================================================================

def get_slope(lat, lon):
    """
    Fetches elevation at 8 points around (lat, lon) - one in each compass
    direction - plus the center point, and returns the STEEPEST gradient
    found among them (in degrees).

    Why 8 directions instead of just north+east (the original approach)?
    Open-Meteo's elevation data is a 90m-resolution DEM (Copernicus GLO-90).
    A real slope failure - a road cut, a river-cut valley wall, an urban
    hillside - is often a narrow feature running in one specific direction.
    If that direction happens to be, say, northwest-to-southeast, a
    north/east-only gradient can average right past it and report a much
    gentler slope than what's actually on the ground. Checking all 8
    compass directions and keeping the steepest one is far less likely to
    miss the real cliff/scarp direction.

    NOTE: even with this fix, a 90m DEM still can't resolve very small
    (tens-of-metres-wide) engineered slopes precisely. If you find this
    still under-reports slope for known steep sites, the next step up is
    swapping in a higher-resolution DEM (e.g. Bhuvan/ISRO CartoDEM at
    ~30m, or ALOS PALSAR at ~12.5m) - see the chat explanation for details.
    """
    offset = 0.001  # ~111 m

    # (delta_lat, delta_lon) for the center point plus all 8 compass directions
    directions = [
        (0, 0),                     # 0: center
        ( offset,  0),              # 1: N
        (-offset,  0),              # 2: S
        ( 0,  offset),              # 3: E
        ( 0, -offset),              # 4: W
        ( offset,  offset),         # 5: NE
        ( offset, -offset),         # 6: NW
        (-offset,  offset),         # 7: SE
        (-offset, -offset),         # 8: SW
    ]
    lats = ",".join(str(lat + d[0]) for d in directions)
    lons = ",".join(str(lon + d[1]) for d in directions)

    url = (
        "https://api.open-meteo.com/v1/elevation"
        f"?latitude={lats}&longitude={lons}"
    )

    elev = requests.get(url, timeout=10).json().get("elevation")

    # Fail LOUD, not quiet: the original code defaulted to [0, 0, 0] on any
    # problem, which silently reports "flat ground" (i.e. LOW risk) for a
    # hazard app whenever the elevation service hiccups. Raising here means
    # the existing try/except in the button handler shows a red st.error
    # instead of quietly under-reporting risk.
    if not elev or len(elev) != 9 or any(e is None for e in elev):
        raise RuntimeError("Elevation service returned incomplete data for this location.")

    base = elev[0]
    steepest = 0.0
    for i, (dlat, dlon) in enumerate(directions[1:], start=1):
        # Real-world horizontal distance (metres) covered by this sample point
        dy = dlat * 111320
        dx = dlon * 111320 * math.cos(math.radians(lat))
        horiz_dist = math.sqrt(dy**2 + dx**2)
        if horiz_dist == 0:
            continue
        rise = elev[i] - base
        angle_deg = math.degrees(math.atan(abs(rise) / horiz_dist))
        steepest = max(steepest, angle_deg)

    return round(steepest, 2)


def get_weather(lat, lon):
    """
    Fetches live weather data from Open-Meteo API.
    Returns temperature (°C), soil moisture (%), rainfall (mm), and rain category.
    """
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&current=temperature_2m,rain,soil_moisture_3_to_9cm"
    )

    # Call the API - .json() converts the text response into a Python dict
    # ["current"] picks the "current conditions" section
    data = requests.get(url, timeout=10).json()["current"]

    rain = data["rain"]  # rainfall in mm/hour

    # Convert the number to a text category that the model was trained on
    if   rain == 0:    category = "clear"
    elif rain < 2.5:   category = "light_rain"
    elif rain < 7.6:   category = "rain"
    elif rain < 50:    category = "continuous_rain"
    else:              category = "downpour"

    return {
        "temperature":   data["temperature_2m"],
        "soil_moisture": data["soil_moisture_3_to_9cm"] * 100,  # m3/m3 -> %
        "rainfall_mm":   rain,
        "category":      category,
    }


def get_recent_rainfall(lat, lon, days=3):
    """
    Fetches TOTAL rainfall over the past `days` days (not just this instant).

    Why this matters: landslides are very often triggered by rain that fell
    HOURS OR DAYS before the slope actually gives way - the ground needs
    time to soak up water and become saturated first. get_weather() above
    only reports what's falling right this second. If you check a location
    the morning after 3 days of heavy rain finally stopped, get_weather()
    will happily report "clear" even though the slope may be at its most
    dangerous point. This function fills that gap.

    Returns a dict with total_mm (float or None if the API call failed)
    and a list of each day's rainfall for transparency in the debug panel.
    """
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&daily=precipitation_sum&past_days={days}&forecast_days=1&timezone=auto"
    )
    try:
        daily = requests.get(url, timeout=10).json()["daily"]
        # precipitation_sum includes TODAY as the last entry, which is a
        # partial/incomplete day - drop it so we only sum full past days
        past_days_mm = daily["precipitation_sum"][:-1]
        return {
            "total_mm": round(sum(v for v in past_days_mm if v is not None), 1),
            "daily_mm": [round(v, 1) if v is not None else None for v in past_days_mm],
        }
    except Exception:
        return {"total_mm": None, "daily_mm": []}


def score_to_label(score):
    """
    Converts a 0-100 risk score into three things:
      1. A text label  ("Low", "Moderate", "High", "Very High")
      2. A banner type ("success", "info", "warning", "error")
         - This tells Streamlit what colour to make the banner
      3. A coloured circle emoji
    """
    if score >= 80: return "Very High", "error",   "\U0001f534"  # red circle
    if score >= 60: return "High",      "warning",  "\U0001f7e0"  # orange circle
    if score >= 30: return "Moderate",  "info",     "\U0001f7e1"  # yellow circle
    return              "Low",       "success",  "\U0001f7e2"  # green circle


def marker_color(score):
    """Returns a map pin colour string for Folium based on the risk score."""
    if score >= 80: return "darkred"
    if score >= 60: return "red"
    if score >= 30: return "orange"
    return "green"


# -----------------------------------------------------------------------------
# SAFE ROUTE ADVISORY HELPERS
# -----------------------------------------------------------------------------

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculates geodesic distance between two GPS coordinates in kilometers."""
    R = 6371.0  # Earth radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


@st.cache_data(ttl=3600, show_spinner=False)
def get_osrm_route(points_tuple):
    """
    Calls Open Source Routing Machine (OSRM) API for road driving directions.
    points_tuple: tuple of (lat, lon) coordinates e.g. ((lat1, lon1), (lat2, lon2))
    Returns dict with distance_km, duration_min, and list of [lat, lon] points.
    """
    coord_str = ";".join(f"{lon:.5f},{lat:.5f}" for lat, lon in points_tuple)
    url = f"http://router.project-osrm.org/route/v1/driving/{coord_str}?overview=full&geometries=geojson"
    try:
        res = requests.get(url, timeout=5).json()
        if res.get("code") == "Ok":
            r0 = res["routes"][0]
            pts = [(pt[1], pt[0]) for pt in r0["geometry"]["coordinates"]]
            return {
                "distance_km": round(r0["distance"] / 1000.0, 2),
                "duration_min": round(r0["duration"] / 60.0, 1),
                "coords": pts
            }
    except Exception:
        pass
    return None


def calculate_safe_route(start, dest, hazard, danger_radius_km=1.5):
    """
    Computes a driving route avoiding active landslide danger zones.
    1. First queries the direct route.
    2. If direct route intersects the hazard radius, tests candidate detour corridors.
    3. Returns safest optimal detour route.
    """
    direct = get_osrm_route((start, dest))
    
    # Offline fallback: straight line geometry if network times out
    if not direct:
        direct = {
            "distance_km": round(haversine_distance(start[0], start[1], dest[0], dest[1]), 2),
            "duration_min": round(haversine_distance(start[0], start[1], dest[0], dest[1]) / 35.0 * 60, 1),
            "coords": [list(start), list(dest)]
        }
    
    # Measure minimum clearance from direct route to hazard center
    min_dist_direct = min(haversine_distance(p[0], p[1], hazard[0], hazard[1]) for p in direct["coords"])
    
    # If already safely clear of danger radius
    if min_dist_direct >= danger_radius_km:
        return {
            "safe_route": direct,
            "is_detour": False,
            "min_hazard_dist": round(min_dist_direct, 2),
            "direct_route": direct,
            "waypoint": None,
            "is_limited_corridor": False
        }
    
    # Direct road intersects danger perimeter: compute detour candidate waypoints
    dy = dest[0] - start[0]
    dx = (dest[1] - start[1]) * math.cos(math.radians(hazard[0]))
    bearing = math.atan2(dy, dx) if (abs(dx) > 1e-6 or abs(dy) > 1e-6) else 0.0
    
    candidate_angles = [
        bearing + math.pi / 2,
        bearing - math.pi / 2,
        bearing + math.pi / 4,
        bearing - math.pi / 4,
        0.0, math.pi, math.pi / 2, 3 * math.pi / 2
    ]
    
    safe_candidates = []
    fallback_candidates = []
    
    for dist_factor in [1.6, 2.4]:
        d_km = danger_radius_km * dist_factor
        for ang in candidate_angles:
            dlat = (d_km * math.sin(ang)) / 111.0
            dlon = (d_km * math.cos(ang)) / (111.0 * math.cos(math.radians(hazard[0])))
            wp = (round(hazard[0] + dlat, 4), round(hazard[1] + dlon, 4))
            detour = get_osrm_route((start, wp, dest))
            if detour:
                m_h = min(haversine_distance(p[0], p[1], hazard[0], hazard[1]) for p in detour["coords"])
                if m_h >= danger_radius_km:
                    safe_candidates.append((detour["distance_km"], m_h, detour, wp))
                else:
                    fallback_candidates.append((m_h, detour["distance_km"], detour, wp))
        if safe_candidates:
            break
            
    if safe_candidates:
        safe_candidates.sort(key=lambda x: x[0])  # prioritize shortest safe distance
        best = safe_candidates[0]
        return {
            "safe_route": best[2],
            "is_detour": True,
            "min_hazard_dist": round(best[1], 2),
            "direct_route": direct,
            "waypoint": best[3],
            "is_limited_corridor": False
        }
    elif fallback_candidates:
        fallback_candidates.sort(key=lambda x: -x[0])  # maximize clearance
        best = fallback_candidates[0]
        return {
            "safe_route": best[2],
            "is_detour": True,
            "min_hazard_dist": round(best[0], 2),
            "direct_route": direct,
            "waypoint": best[3],
            "is_limited_corridor": True
        }
    
    return {
        "safe_route": direct,
        "is_detour": False,
        "min_hazard_dist": round(min_dist_direct, 2),
        "direct_route": direct,
        "waypoint": None,
        "is_limited_corridor": True
    }


# =============================================================================
# PART 5 - SIDEBAR  (the left panel)
# Everything inside "with st.sidebar:" appears on the left side of the screen.
# =============================================================================

with st.sidebar:

    st.image("http://img.lrp.pp.ua/u/36n9pa.png", width=100, use_container_width=False)

    st.markdown(
    '<h1 style="margin-top: 0px;"> ⛰️BhuAlert</h1>',
    unsafe_allow_html=True
    )

    st.caption("Landslide Early Warning \u00b7 SIH Prototype")

    st.caption("Trained on 351 NER landslide records")

    # If the live rain categories don't match what the CSV used for training,
    # rain is silently having zero effect on every prediction - see the
    # sanity check right after train_model() above.

    st.divider()

    # Number input widgets - user types or clicks +/- to change values
    lat = st.number_input("Latitude",  value=25.6700,max_value=90.0000,min_value=-90.0000, format="%.4f", step=0.001)
    lon = st.number_input("Longitude", value=94.1100,max_value=180.0000,min_value=-180.0000, format="%.4f", step=0.001)

    # A button that returns True the moment it is clicked, False otherwise
    go = st.button("\u26a1 Run Assessment", type="primary", use_container_width=True)


# =============================================================================
# PART 6 - MAIN PAGE HEADER
# These lines appear at the top of the main (right) area.
# =============================================================================

st.title("\u26f0\ufe0f Landslide Risk Assessment")
st.caption("Enter coordinates on the left, Then hit Run Assessment.")
st.divider()


# =============================================================================
# PART 7 - HANDLE BUTTON CLICK
# "if go:" is True only in the exact moment the user clicks the button.
# Everything inside runs, then Streamlit re-renders the page.
# =============================================================================

if go:
    # st.spinner shows a loading animation while the code inside runs
    with st.spinner("Fetching live terrain & weather data..."):
        try:
            # Step 1: call the Open-Meteo elevation API to get slope
            slope = get_slope(lat, lon)

            # Step 2: call the Open-Meteo forecast API to get weather
            weather = get_weather(lat, lon)

            # Step 2b: also pull the last 3 days of rainfall. See
            # get_recent_rainfall()'s docstring - this is what lets the app
            # recognise a spot that got soaked by rain yesterday even if
            # it's perfectly dry and sunny at the exact moment you check it.
            recent_rain = get_recent_rainfall(lat, lon, days=3)

            # Step 3: put the 4 inputs into a one-row DataFrame (table)
            # The model was trained on a table, so it expects a table as input
            features = pd.DataFrame([{
                "inclination":           slope,
                "soil_moisture":         weather["soil_moisture"],
                "temperature":           weather["temperature"],
                "rain_report (trigger)": weather["category"],
            }])

            # Step 4: run the model
            # predict_proba returns [[prob_of_0, prob_of_1]]
            # [0] = first (only) row
            # [1] = second value = probability the location IS at risk
            # Multiply by 100 to get a 0-100 percentage
            ml_score = int(model.predict_proba(features)[0][1] * 100)

            # Step 4b: antecedent-rainfall adjustment.
            # The trained model only ever "sees" the CURRENT hour's rain
            # (that's all it was trained on). This adds a small, clearly
            # labelled rule-based nudge on top when the past few days were
            # genuinely wet, so a recently-soaked slope isn't reported as
            # perfectly safe just because it happens to be dry right now.
            # These thresholds are a reasonable starting point, not a
            # calibrated scientific figure - tune them if/when you have
            # real historical rainfall-vs-landslide data for NER.
            antecedent_bonus = 0
            if recent_rain["total_mm"] is not None:
                if recent_rain["total_mm"] >= 150:
                    antecedent_bonus = 25
                elif recent_rain["total_mm"] >= 75:
                    antecedent_bonus = 15
                elif recent_rain["total_mm"] >= 30:
                    antecedent_bonus = 5

            risk_score = min(100, ml_score + antecedent_bonus)

            # Step 5: save results to session_state
            # Streamlit re-runs the whole file after every interaction.
            # session_state is a dictionary that SURVIVES re-runs, so the
            # results stay visible even after the page re-renders.
            st.session_state.result = {
                "lat":       lat,
                "lon":       lon,
                "slope":     slope,
                **weather,           # unpacks all weather keys into this dict
                "ml_score":         ml_score,
                "antecedent_bonus": antecedent_bonus,
                "recent_rain_mm":   recent_rain["total_mm"],
                "recent_rain_daily": recent_rain["daily_mm"],
                "risk_score":       risk_score,
            }

        except Exception as e:
            # If anything fails (API down, bad data etc.), show a red error box
            st.error(f"Something went wrong: {e}")


# =============================================================================
# PART 8 - DISPLAY RESULTS
# This runs on EVERY re-render, not just when button is clicked.
# It checks if we have saved results in session_state and shows them.
# =============================================================================

if "result" in st.session_state and st.session_state.result:

    # Pull the saved results out of session_state
    r     = st.session_state.result
    score = r["risk_score"]

    # Get the label, banner colour type, and emoji for this score
    label, alert_type, emoji = score_to_label(score)

    # ---- Risk banner ----
    # getattr(st, "error") is the same as calling st.error(...)
    # We use getattr because alert_type is a variable, not a fixed word
    getattr(st, alert_type)(f"{emoji} {label.upper()} RISK  |  Score: {score} / 100")

    # ---- 5 metric cards in a row ----
    # st.columns(5) creates 5 equal-width columns side by side
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Risk Score",    f"{score}/100")
    c2.metric("Risk Level",    f"{emoji} {label}")
    c3.metric("Slope",         f"{r['slope']}\u00b0")
    c4.metric("Temperature",   f"{r['temperature']} \u00b0C")
    c5.metric("Soil Moisture", f"{r['soil_moisture']:.1f}%")

    # ---- Two-column layout: map on left (wider), details table on right ----
    # [2, 1] means left column is twice as wide as right column
    col_map, col_info = st.columns([2, 1])

    with col_map:
        st.subheader("\U0001f5fa\ufe0f Location Map")

        # Create a Folium map centred on the user's coordinates
        # tiles = the background map imagery (Esri satellite, free, no API key)
        m = folium.Map(
            location=[r["lat"], r["lon"]],
            zoom_start=14,
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri"
        )

        # Drop a coloured pin at the exact coordinates
        folium.Marker(
            location=[r["lat"], r["lon"]],
            popup=f"Risk: {score}% ({label})",            # text shown on click
            icon=folium.Icon(color=marker_color(score), icon="exclamation-sign"),
        ).add_to(m)

        # Draw a translucent circle showing the 1.5km risk zone around the point
        folium.Circle(
            location=[r["lat"], r["lon"]],
            radius=1500,   # metres
            color="#ef4444" if score >= 60 else "#f59e0b" if score >= 30 else "#22c55e",
            fill=True,
            fill_opacity=0.15,
        ).add_to(m)

        # Embed the Folium map into Streamlit
        # returned_objects=[] prevents the map from sending click data back to Python
        st_folium(m, height=400, use_container_width=True, returned_objects=[])

    with col_info:
        st.subheader("\U0001f4cb Details")

        # st.markdown renders text as Markdown (| | creates a table)
        st.markdown(f"""
| Field | Value |
|-------|-------|
| Latitude | `{r["lat"]:.4f}` |
| Longitude | `{r["lon"]:.4f}` |
| Slope | `{r["slope"]}\u00b0` |
| Temperature | `{r["temperature"]} \u00b0C` |
| Soil Moisture | `{r["soil_moisture"]:.1f}%` |
| Rainfall | `{r["rainfall_mm"]} mm` |
| Rain Type | `{r["category"]}` |
| Risk Score | `{score} / 100` |
""")

    # ---- Debug panel: exactly what went into the score, and why ----
    # This is the single most useful panel for answering "why is this
    # score low/high?" for any coordinate - it shows each of the three
    # things the model needs to see together (steep + wet + rainy) before
    # it will call a spot high-risk, plus the antecedent-rainfall nudge.
    with st.expander("\U0001f50d Why this score? (model inputs)"):
        # Built as a separate plain string first (not inlined in the f-string
        # below) so this stays compatible with Python versions older than
        # 3.12, which don't allow nested f-strings using the same quotes.
        daily_list = r.get("recent_rain_daily") or []
        daily_breakdown = ""
        if daily_list:
            daily_parts = [f"{v} mm" for v in daily_list]
            daily_breakdown = " (daily: " + ", ".join(daily_parts) + ")"

        st.markdown(f"""
- **Computed slope:** `{r['slope']}°` — only adds points once it's over **35°** (extra points over **50°**)
- **Current soil moisture:** `{r['soil_moisture']:.1f}%` — only adds points once it's over **75%**
- **Current rain:** `{r['category']}` (`{r['rainfall_mm']} mm/hr` right now)
- **Rainfall, past 3 days:** `{r['recent_rain_mm']} mm`{daily_breakdown}
- **Live-conditions model score:** `{r['ml_score']}/100`
- **Antecedent-rain adjustment:** `+{r['antecedent_bonus']}`
- **Final score shown above:** `{score}/100`
""")
        st.caption(
            "The underlying model was trained to require a steep slope AND "
            "wet/rainy conditions at the same time. A historically active "
            "landslide site can still show up as low risk here if you check "
            "it on a dry day, the computed slope comes out gentler than the "
            "real on-the-ground terrain (elevation data has ~90m resolution "
            "and can miss narrow features), or the past-3-day rain wasn't "
            "enough to trigger the adjustment above."
        )

    # ---- Emergency contacts (only shown when risk is High or Very High) ----
    if score >= 30:
        st.divider()
        st.subheader("\U0001f6a8 Emergency Contacts")
        e1, e2, e3 = st.columns(3)
        e1.metric("Emergency",    "112")
        e2.metric("Ambulance",    "108")
        e3.metric("Fire & Rescue", "101")

        # =============================================================================
    # SAFE ROUTE ADVISORY (LANDSLIDE AVOIDANCE)
    # =============================================================================
    st.divider()
    st.subheader("🧭 Safe Route Advisory")
    st.caption("Plan a safe navigation route avoiding the active landslide hazard zone.")

    if score >=30:

        # Initialize default route coordinates based on the assessed landslide location
        default_start_lat = round(r["lat"] - 0.0250, 4)
        default_start_lon = round(r["lon"] - 0.0200, 4)
        default_dest_lat  = round(r["lat"] + 0.0250, 4)
        default_dest_lon  = round(r["lon"] + 0.0200, 4)

        # Input columns for Start (Current Location) and Destination
        col_start, col_dest = st.columns(2)
        
        with col_start:
            st.markdown("**📍 Current Location (Start)**")
            start_lat = st.number_input("Start Latitude", value=default_start_lat, format="%.4f", step=0.001, key="start_lat_key")
            start_lon = st.number_input("Start Longitude", value=default_start_lon, format="%.4f", step=0.001, key="start_lon_key")

        with col_dest:
            st.markdown("**🎯 Destination (End)**")
            dest_lat = st.number_input("Destination Latitude", value=default_dest_lat, format="%.4f", step=0.001, key="dest_lat_key")
            dest_lon = st.number_input("Destination Longitude", value=default_dest_lon, format="%.4f", step=0.001, key="dest_lon_key")

        col_buf, col_btn = st.columns([2, 1])
        with col_buf:
            danger_buf_km = st.slider(
                "Landslide Safety Clearance Buffer (km)",
                min_value=0.5,
                max_value=3.0,
                value=1.5,
                step=0.1,
                help="Minimum safety clearance radius maintained around the active landslide zone."
            )
        with col_btn:
            st.write("")
            st.write("")
            calc_route = st.button("🚗 Recalculate Route", type="primary", use_container_width=True)

        # Compute route
        with st.spinner("Analyzing road network & computing safe bypass route..."):
            hazard_pt = (r["lat"], r["lon"])
            start_pt  = (start_lat, start_lon)
            dest_pt   = (dest_lat, dest_lon)
            
            route_data = calculate_safe_route(start_pt, dest_pt, hazard_pt, danger_radius_km=danger_buf_km)

        if route_data and route_data.get("safe_route"):
            safe_r   = route_data["safe_route"]
            direct_r = route_data["direct_route"]
            is_detour = route_data["is_detour"]
            min_clearance = route_data["min_hazard_dist"]
            is_limited = route_data.get("is_limited_corridor", False)

            # Status Alert Banner
            if is_detour and not is_limited:
                extra_km = round(safe_r["distance_km"] - direct_r["distance_km"], 1)
                extra_min = round(safe_r["duration_min"] - direct_r["duration_min"], 1)
                st.success(
                    f"🛡️ **SAFE DETOUR ACTIVE:** Direct route passes within {route_data['direct_route']['distance_km']} km of the active landslide danger perimeter. "
                    f"Advised safe bypass (+{extra_km} km, +{extra_min} mins) maintains a safe {min_clearance:.2f} km clearance."
                )
            elif is_limited:
                st.warning(
                    f"⚠️ **LIMITED MOUNTAIN PASS CORRIDOR:** Alternate bypass roads are restricted in this valley. "
                    f"Route passes {min_clearance:.2f} km from the hazard center. Drive with extreme vigilance and check BRO updates."
                )
            else:
                st.success(
                    f"✅ **DIRECT ROUTE IS CLEAR & SAFE:** Road maintains a safe {min_clearance:.2f} km clearance from the landslide hazard zone."
                )

            # 4 Metric Cards
            m1, m2, m3, m4 = st.columns(4)
            m1.metric(
                "Route Distance",
                f"{safe_r['distance_km']} km",
                delta=f"+{round(safe_r['distance_km'] - direct_r['distance_km'], 1)} km detour" if is_detour else "Direct"
            )
            m2.metric(
                "Est. Travel Time",
                f"{safe_r['duration_min']} mins",
                delta=f"+{round(safe_r['duration_min'] - direct_r['duration_min'], 1)} mins" if is_detour else "Optimal"
            )
            m3.metric(
                "Hazard Clearance",
                f"{min_clearance:.2f} km",
                delta="Safe Buffer" if min_clearance >= danger_buf_km else "Caution"
            )
            m4.metric(
                "Advisory Status",
                "🛡️ Detour Active" if is_detour else "✅ Route Safe"
            )

            # Map Center & Bounds
            all_pts = safe_r["coords"] + [list(hazard_pt), list(start_pt), list(dest_pt)]
            lats = [p[0] for p in all_pts]
            lons = [p[1] for p in all_pts]
            center_lat = sum(lats) / len(lats)
            center_lon = sum(lons) / len(lons)

            # Google Maps style Folium Map
            m_route = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=12,
                tiles="https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}",
                attr="Google Maps"
            )
            m_route.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]], padding=[35, 35])

            # 1. Active Landslide Hazard Zone (Red translucent circle)
            danger_radius_m = int(danger_buf_km * 1000)
            folium.Circle(
                location=hazard_pt,
                radius=danger_radius_m,
                color="#d93025",
                weight=2,
                fill=True,
                fill_color="#ea4335",
                fill_opacity=0.22,
                dash_array="5, 5",
                popup=folium.Popup(f"<b>⚠️ Active Landslide Zone</b><br>Danger Radius: {danger_buf_km} km<br>Assessed Risk: {score}/100", max_width=250),
                tooltip="⚠️ Active Landslide Hazard Zone"
            ).add_to(m_route)

            folium.Marker(
                location=hazard_pt,
                popup=f"<b>⚠️ Landslide Epicenter (Risk: {score}/100)</b>",
                tooltip="Landslide Risk Point",
                icon=folium.Icon(color="darkred", icon="warning-sign")
            ).add_to(m_route)

            # 2. Blocked Direct Road (if detour active)
            if is_detour and direct_r and direct_r.get("coords"):
                folium.PolyLine(
                    locations=direct_r["coords"],
                    color="#ea4335",
                    weight=4,
                    opacity=0.75,
                    dash_array="6, 8",
                    tooltip="🚫 Blocked Direct Route (Passes through landslide zone)"
                ).add_to(m_route)

            # 3. Google Maps Style Recommended Safe Route (Blue line with white casing)
            folium.PolyLine(
                locations=safe_r["coords"],
                color="#ffffff",
                weight=8,
                opacity=0.85
            ).add_to(m_route)

            folium.PolyLine(
                locations=safe_r["coords"],
                color="#1a73e8",
                weight=5,
                opacity=0.95,
                tooltip="🛡️ Advised Safe Route"
            ).add_to(m_route)

            # 4. Detour Waypoint Marker (if detour active)
            if route_data.get("waypoint"):
                folium.CircleMarker(
                    location=route_data["waypoint"],
                    radius=6,
                    color="#ffffff",
                    weight=2,
                    fill=True,
                    fill_color="#34a853",
                    fill_opacity=1.0,
                    popup=folium.Popup("<b>🛡️ Safe Detour Waypoint</b>", max_width=200),
                    tooltip="Safe Detour Waypoint"
                ).add_to(m_route)

            # 5. Current Location / Origin (Google Maps blue pulsing GPS dot)
            folium.CircleMarker(
                location=start_pt,
                radius=13,
                color="#4285f4",
                weight=1.5,
                fill=True,
                fill_color="#4285f4",
                fill_opacity=0.25
            ).add_to(m_route)

            folium.CircleMarker(
                location=start_pt,
                radius=7,
                color="#ffffff",
                weight=3,
                fill=True,
                fill_color="#1a73e8",
                fill_opacity=1.0,
                popup=folium.Popup("<b>📍 Current Location (Start)</b>", max_width=200),
                tooltip="Current Location (Start)"
            ).add_to(m_route)

            # 6. Target Destination (Classic Red Marker Pin)
            folium.Marker(
                location=dest_pt,
                popup=folium.Popup("<b>🎯 Target Destination</b>", max_width=200),
                tooltip="Target Destination",
                icon=folium.Icon(color="red", icon="flag")
            ).add_to(m_route)

            # Embed Google Maps style Folium Map
            st_folium(m_route, height=460, use_container_width=True, returned_objects=[], key="safe_route_map")

            # Advisory Guidance
            st.info(
                "💡 **Navigation Advisory:** The bold blue line illustrates the recommended safe road corridor. "
                + ("The dashed red line indicates the hazardous direct segment to avoid. " if is_detour else "")
                + "Always avoid night driving during active monsoons and verify route conditions with local authorities."
            )
        else:
            st.error("Could not calculate a route between the specified coordinates. Please adjust the coordinates.")


    else:
        # Shown before the user clicks the button for the first time
        st.info("🍀Your location is safe, No evacuation routes required.")
