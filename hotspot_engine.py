"""
Hotspot engine: cleaning, density-based hotspot detection, risk scoring,
time-window weighting and patrol-unit allocation.

Pure data/ML code with no UI, so it can be tested or reused in a script.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN

# Study-area bounds (Bengaluru). Records outside this box are rejected as bad coordinates.
BBOX = {"lat_min": 12.70, "lat_max": 13.30, "lon_min": 77.30, "lon_max": 78.00}

# Relative severity weights. Adjust to match your force's offence classification.
SEVERITY = {
    "homicide": 10, "murder": 10,
    "armed robbery": 5, "assault": 5, "robbery": 4,
    "burglary": 3, "house break-in": 3, "vehicle theft": 3, "chain snatching": 3,
    "theft": 2, "harassment": 2,
    "fraud": 1, "vandalism": 1, "other": 1,
}

# Column names we try to find automatically (after lowercasing and replacing spaces with _).
COLUMN_ALIASES = {
    "lat": ["latitude", "lat", "y"],
    "lon": ["longitude", "lon", "lng", "long", "x"],
    "time": ["datetime", "date_time", "timestamp", "occurred_at", "occurrence_time", "date", "time"],
    "type": ["crime_type", "type", "offence", "offense", "category", "crime"],
    "area": ["area", "locality", "neighbourhood", "neighborhood", "police_station", "station"],
}


def _norm(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_")


def detect_columns(df: pd.DataFrame) -> dict:
    """Map logical fields (lat, lon, time, type, area) to the uploaded file's column names."""
    lookup = {_norm(c): c for c in df.columns}
    mapping = {}
    for key, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in lookup:
                mapping[key] = lookup[alias]
                break
    return mapping


def haversine_m(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance in metres."""
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype=float)) for v in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371000.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def peak_window(hours: np.ndarray) -> str:
    """Return the 3-hour block that contains the most incidents, e.g. '20:00-23:00'."""
    counts = np.bincount(hours.astype(int), minlength=24)
    best = max(range(24), key=lambda s: counts[[(s + k) % 24 for k in range(3)]].sum())
    return f"{best:02d}:00-{(best + 3) % 24:02d}:00"


def clean_incidents(raw: pd.DataFrame, mapping: dict, bbox: dict = BBOX) -> tuple[pd.DataFrame, dict]:
    """Standardise columns, drop unusable rows and report what was removed."""
    report = {"rows_in": len(raw)}
    df = pd.DataFrame(index=raw.index)
    df["lat"] = pd.to_numeric(raw[mapping["lat"]], errors="coerce")
    df["lon"] = pd.to_numeric(raw[mapping["lon"]], errors="coerce")
    df["ts"] = pd.to_datetime(raw[mapping["time"]], errors="coerce", dayfirst=True)
    df["crime_type"] = (
        raw[mapping["type"]].astype(str).str.strip().str.lower() if "type" in mapping else "other"
    )
    df["area"] = (
        raw[mapping["area"]].astype(str).str.strip().str.title() if "area" in mapping else "Unknown"
    )

    bad_coord = df[["lat", "lon"]].isna().any(axis=1)
    in_box = df.lat.between(bbox["lat_min"], bbox["lat_max"]) & df.lon.between(bbox["lon_min"], bbox["lon_max"])
    bad_time = df.ts.isna()

    report["missing_or_invalid_coords"] = int(bad_coord.sum())
    report["outside_study_area"] = int((~bad_coord & ~in_box).sum())
    report["bad_timestamp"] = int((~bad_coord & in_box & bad_time).sum())

    df = df[~bad_coord & in_box & ~bad_time].copy()
    before = len(df)
    df = df.drop_duplicates(subset=["lat", "lon", "ts", "crime_type"])
    report["duplicates_removed"] = before - len(df)
    report["rows_kept"] = len(df)

    df["hour"] = df.ts.dt.hour
    df["weekday"] = df.ts.dt.day_name()
    df["month"] = df.ts.dt.to_period("M").astype(str)
    df["severity"] = df.crime_type.map(SEVERITY).fillna(1.0)
    return df.reset_index(drop=True), report


def add_recency_weight(df: pd.DataFrame, half_life_days: float = 180) -> pd.DataFrame:
    """Recent incidents count more: weight halves every `half_life_days`."""
    df = df.copy()
    age_days = (df.ts.max() - df.ts).dt.total_seconds() / 86400
    df["recency"] = 0.5 ** (age_days / half_life_days)
    df["weight"] = df.severity * df.recency
    return df


def detect_hotspots(
    df: pd.DataFrame,
    min_cluster_size: int = 15,
    min_samples: int = 5,
    half_life_days: float = 180,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Cluster incidents with HDBSCAN on great-circle distance.
    Returns (hotspot summary table, incident table labelled with cluster ids).
    Cluster -1 means the incident is not part of any hotspot.
    """
    df = add_recency_weight(df, half_life_days)
    coords = np.radians(df[["lat", "lon"]].to_numpy())
    labels = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="haversine",
        cluster_selection_method="eom",
    ).fit_predict(coords)
    df["cluster"] = labels

    ref = df.ts.max()
    rows = []
    for cid, g in df[df.cluster >= 0].groupby("cluster"):
        c_lat, c_lon = g.lat.mean(), g.lon.mean()
        dist = haversine_m(g.lat, g.lon, c_lat, c_lon)
        rows.append({
            "cluster": int(cid),
            "lat": c_lat,
            "lon": c_lon,
            "radius_m": int(max(np.percentile(dist, 90), 150)),
            "incidents": len(g),
            "last_30d": int((g.ts >= ref - pd.Timedelta(days=30)).sum()),
            "weighted_risk": g.weight.sum(),
            "top_crime": g.crime_type.value_counts().idxmax(),
            "top_area": g.area.mode().iloc[0],
            "peak_window": peak_window(g.hour.to_numpy()),
        })

    hs = pd.DataFrame(rows)
    if hs.empty:
        return hs, df

    hs["risk_score"] = (100 * hs.weighted_risk / hs.weighted_risk.max()).round(1)
    hs = hs.sort_values("risk_score", ascending=False).reset_index(drop=True)
    hs.insert(0, "hotspot_id", [f"H{i + 1:02d}" for i in range(len(hs))])
    return hs, df


def apply_time_window(labeled: pd.DataFrame, hs: pd.DataFrame, start_hour: int, end_hour: int) -> pd.DataFrame:
    """Score each hotspot only on incidents inside the chosen hour window (wraps past midnight)."""
    hs = hs.copy()
    if hs.empty:
        hs["window_score"] = pd.Series(dtype=float)
        hs["window_incidents"] = pd.Series(dtype=int)
        return hs

    h = labeled.hour
    if start_hour <= end_hour:
        mask = h.between(start_hour, end_hour)
    else:
        mask = (h >= start_hour) | (h <= end_hour)
    in_window = labeled[mask & (labeled.cluster >= 0)]

    raw = hs.cluster.map(in_window.groupby("cluster").weight.sum()).fillna(0.0)
    top = raw.max()
    hs["window_score"] = (100 * raw / top).round(1) if top > 0 else 0.0
    hs["window_incidents"] = hs.cluster.map(in_window.groupby("cluster").size()).fillna(0).astype(int)
    return hs


def allocate_units(hs: pd.DataFrame, units: int, score_col: str = "window_score") -> pd.DataFrame:
    """
    Split available patrol units across hotspots using D'Hondt apportionment:
    each unit goes to the hotspot with the highest score / (units already given + 1).
    This favours high-risk areas while still spreading units across hotspots.
    """
    hs = hs.copy()
    hs["units"] = 0
    if hs.empty or units <= 0:
        return hs

    scores = hs[score_col].to_numpy(dtype=float)
    if scores.sum() <= 0:
        return hs

    alloc = np.zeros(len(hs), dtype=int)
    for _ in range(int(units)):
        alloc[np.argmax(scores / (alloc + 1))] += 1
    hs["units"] = alloc
    return hs.sort_values(["units", score_col], ascending=False).reset_index(drop=True)
