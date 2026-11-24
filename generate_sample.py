"""
Generates a SYNTHETIC incident file around Koramangala, Whitefield and Indiranagar
so you can test the dashboard. Replace it with your real cleaned dataset.

Run:  python generate_sample.py  ->  incidents_sample.csv
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

AREAS = {
    "Koramangala": (12.9352, 77.6245),
    "Whitefield": (12.9698, 77.7500),
    "Indiranagar": (12.9719, 77.6412),
}
CRIME_TYPES = ["theft", "chain snatching", "vehicle theft", "burglary", "robbery",
               "assault", "harassment", "fraud", "vandalism"]
CRIME_P = [0.28, 0.12, 0.10, 0.10, 0.07, 0.06, 0.10, 0.09, 0.08]
END = pd.Timestamp("2026-06-30")

rows = []
for area, (lat, lon) in AREAS.items():
    for _ in range(4):  # four hidden hotspot centres per area
        c_lat = lat + rng.normal(0, 0.008)
        c_lon = lon + rng.normal(0, 0.008)
        n = int(rng.integers(60, 160))
        days = rng.integers(0, 540, n)
        # evening-heavy hours
        hours = np.where(rng.random(n) < 0.6, rng.integers(18, 24, n), rng.integers(0, 24, n))
        minutes = rng.integers(0, 60, n)
        ts = END - pd.to_timedelta(days, unit="D") + pd.to_timedelta(hours, unit="h") + pd.to_timedelta(minutes, unit="m")
        for i in range(n):
            rows.append((
                c_lat + rng.normal(0, 0.0015),
                c_lon + rng.normal(0, 0.0015),
                ts[i],
                rng.choice(CRIME_TYPES, p=CRIME_P),
                area,
            ))

# background noise across all three areas
for _ in range(250):
    area, (lat, lon) = list(AREAS.items())[rng.integers(0, 3)]
    ts = END - pd.Timedelta(days=int(rng.integers(0, 540)), hours=int(rng.integers(0, 24)))
    rows.append((lat + rng.normal(0, 0.012), lon + rng.normal(0, 0.012), ts,
                 rng.choice(CRIME_TYPES, p=CRIME_P), area))

df = pd.DataFrame(rows, columns=["latitude", "longitude", "datetime", "crime_type", "area"])
df["datetime"] = df["datetime"].dt.strftime("%d-%m-%Y %H:%M")
df["latitude"] = df["latitude"].round(6)
df["longitude"] = df["longitude"].round(6)
df.to_csv("incidents_sample.csv", index=False)
print(f"Wrote {len(df)} synthetic incidents to incidents_sample.csv")
