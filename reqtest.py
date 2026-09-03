import requests

params = {
    "latitude": 25.57,   # example: Shillong, Meghalaya
    "longitude": 91.88,
    "hourly": "precipitation",
    "past_days": 7,
    "forecast_days": 3
}
response = requests.get("https://api.open-meteo.com/v1/forecast", params=params)
data = response.json()
print(data["hourly"]["precipitation"])