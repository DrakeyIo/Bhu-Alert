import math
import requests
from datetime import date, timedelta
 
 
def get_rainfall_7day(lat, lon, target_date=None):
    """Sum of rainfall (mm) over the 7 days before target_date. Source: Open-Meteo."""
    if target_date is None:
        target_date = date.today()
    start = target_date - timedelta(days=7)
 
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start.isoformat(),
        "end_date": target_date.isoformat(),
        "daily": "precipitation_sum",
        "timezone": "auto",
    }
    resp = requests.get("https://api.open-meteo.com/v1/forecast", params=params)
    resp.raise_for_status()
    data = resp.json()
    return sum(data["daily"]["precipitation_sum"])

total_Rain = get_rainfall_7day(22.533475,88.348558)

print(total_Rain)