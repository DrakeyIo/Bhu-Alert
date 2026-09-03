import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
 
# ---- 1. Example training data ----
# Each row = one location. label 1 = landslide occurred, 0 = stable point.
data = {
    "slope_deg":          [42, 8,  55, 12, 38, 5,  60, 15, 33, 7],
    "elevation_m":        [1200, 300, 1800, 450, 950, 200, 2100, 500, 1100, 250],
    "ndvi":               [0.25, 0.75, 0.15, 0.68, 0.30, 0.80, 0.10, 0.60, 0.28, 0.72],
    "dist_to_road_m":     [20, 500, 15, 800, 40, 1000, 10, 600, 35, 900],
    "rainfall_7day_mm":   [180, 40, 220, 30, 150, 20, 260, 35, 170, 25],
    "dist_to_drainage_m": [30, 600, 25, 700, 50, 900, 15, 650, 45, 850],
    "label":              [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
}
df = pd.DataFrame(data)
 
# ---- 2. Split features/labels ----
X = df.drop(columns="label")
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=2
)
 
# ---- 3. Train model ----
model = RandomForestClassifier(n_estimators=100, random_state=42)  #n estimators = number of trees in the forest
model.fit(X_train, y_train)
 
# ---- 4. Evaluate ----
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
 
# ---- 5. Predict risk for a new location ----
new_point = pd.DataFrame([{
    "slope_deg": 45,
    "elevation_m": 1300,
    "ndvi": 0.22,
    "dist_to_road_m": 25,
    "rainfall_7day_mm": 190,
    "dist_to_drainage_m": 35,
}])
risk_prob = model.predict_proba(new_point)[0][1]
print(f"Landslide risk probability: {risk_prob:.2%}")


# data = {
#     "slope_deg":          [42, 8,  55, 12, 38, 5,  60, 15, 33, 7],
#     "elevation_m":        [1200, 300, 1800, 450, 950, 200, 2100, 500, 1100, 250],
#     "ndvi":               [0.25, 0.75, 0.15, 0.68, 0.30, 0.80, 0.10, 0.60, 0.28, 0.72],
#     "dist_to_road_m":     [20, 500, 15, 800, 40, 1000, 10, 600, 35, 900],
#     "rainfall_7day_mm":   [180, 40, 220, 30, 150, 20, 260, 35, 170, 25],
#     "dist_to_drainage_m": [30, 600, 25, 700, 50, 900, 15, 650, 45, 850],
#     "label":              [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
# }
# df = pd.DataFrame(data)


# print(df)
# # ---- 2. Split features/labels ----
# X = df.drop(columns="label")

# print(X)