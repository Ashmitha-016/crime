Streamlit dashboard that takes a file of incident records, detects crime hotspots using machine learning, and recommends how to allocate a limited number of patrol units across hotspots and time windows.

The system works on any incident dataset with coordinates and timestamps. It was designed around cleaned records from Koramangala, Whitefield and Indiranagar, Bengaluru.

## Features

- **Upload any CSV or XLSX** and map its columns automatically (with manual override)
- **Automatic data cleaning** with a report of every removed row (invalid coordinates, outside the study area, bad timestamps, duplicates)
- **Hotspot detection** using HDBSCAN on great-circle distance, with no need to fix the number of hotspots in advance
- **Risk scoring** weighted by crime severity and recency
- **Time-window analysis** that re-scores hotspots for any hour range, including windows that cross midnight
- **Patrol allocation** across hotspots using D'Hondt apportionment
- **Satellite map** (Esri World Imagery) with hotspot circles, centroid markers, unit labels, incident points and a density heatmap
- **Dispatch plan** with a risk table, written dispatch orders and CSV export
- **Analytics** for crime types, weekday-by-hour patterns and monthly trends

## Project structure

```
crime-hotspot/
├── app.py                 # Streamlit dashboard (UI)
├── hotspot_engine.py      # Cleaning, clustering, scoring and allocation (no UI code)
├── generate_sample.py     # Creates a synthetic test dataset
├── incidents_sample.csv   # Synthetic test data (not real incidents)
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

Requires Python 3.10 or newer.

```bash
git clone https://github.com/<your-username>/crime-hotspot.git
cd crime-hotspot

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Running the dashboard

```bash
python -m streamlit run app.py
```

Open the address Streamlit prints (usually http://localhost:8501), then upload a dataset in the sidebar.

To try the system without real data, generate the synthetic file and upload `incidents_sample.csv`:

```bash
python generate_sample.py
```

## Input data format

The file needs at least these columns. Column names are matched automatically, ignoring case and spaces.

| Field | Accepted names (examples) | Required |
|---|---|---|
| Latitude | `latitude`, `lat` | Yes |
| Longitude | `longitude`, `lon`, `lng` | Yes |
| Timestamp | `datetime`, `timestamp`, `date`, `occurred_at` | Yes |
| Crime type | `crime_type`, `type`, `offence`, `category` | No (improves output) |
| Area | `area`, `locality`, `police_station` | No (improves output) |

Timestamps are read as day-first (for example `24-11-2026 18:30`).

## Method

1. **Cleaning:** rows with missing or invalid coordinates, coordinates outside the study area, unreadable timestamps and duplicates are removed and counted.
2. **Severity and recency weights:** each incident's weight is its severity score multiplied by a recency factor that halves every `half_life` days (default 180).
3. **Clustering:** HDBSCAN groups incidents by great-circle distance. Incidents that don't belong to a dense group are treated as noise and excluded from hotspots.
4. **Hotspot summary:** each cluster gets a centroid, a radius (90th percentile of distances), incident counts, the top crime type, the most common area and its peak three-hour window.
5. **Risk score:** each hotspot's weighted risk, scaled to 0 to 100.
6. **Time window:** hotspots are re-scored using only incidents inside the chosen hour range.
7. **Allocation:** available units are assigned one at a time to the hotspot with the highest `score / (units already assigned + 1)`.

Parameters you can tune in the sidebar: minimum hotspot size, core density, recency half-life, time window, and units available.

## Configuration

- `BBOX` in `hotspot_engine.py` sets the study-area boundaries. Change it if your data covers a different region.
- `SEVERITY` in `hotspot_engine.py` sets the severity weight for each crime type. Adjust these to match your classification.

## Limitations

- **Reporting bias:** recorded incidents reflect where crimes are reported and where police patrol, not only where crime occurs. Allocation based on recorded data can reinforce existing patterns. Results should be validated against other sources, such as victim surveys.
- **Descriptive, not predictive:** the current version ranks areas with past concentrations. It does not forecast future incidents.
- **Area-level only:** the system contains no personal data and is not designed to assess individuals.
- **Parameter sensitivity:** hotspot results depend on the clustering settings. Check that key hotspots stay stable across reasonable settings.
- **Synthetic sample data:** `incidents_sample.csv` is generated and does not represent real incidents.

## Responsible use

This tool is intended for planning and review by trained personnel. Hotspot outputs should be checked against local knowledge and station records before any deployment decision. They must not be used as grounds for stopping, searching or targeting any individual.

## Roadmap

- Backtesting against held-out incident data
- Forecasting next-period risk per hotspot
- Station and nearest-station mapping for dispatch orders
- Access control and audit logging for multi-user deployment

## License

Add a license before publishing publicly (for example MIT). Choose one at https://choosealicense.com.
