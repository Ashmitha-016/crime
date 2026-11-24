"""
Crime Hotspot & Dispatch Console
Run with:  streamlit run app.py
"""
import io

import folium
import pandas as pd
import plotly.express as px
import streamlit as st
from folium.plugins import Fullscreen, HeatMap
from streamlit_folium import st_folium

from hotspot_engine import (
    allocate_units,
    apply_time_window,
    clean_incidents,
    detect_columns,
    detect_hotspots,
)

ESRI_IMAGERY = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
ESRI_LABELS = "https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}"
ESRI_ATTR = "Imagery © Esri, Maxar, Earthstar Geographics, and the GIS User Community"

PALETTE = ["#38bdf8", "#f472b6", "#a3e635", "#facc15", "#c084fc", "#fb923c", "#2dd4bf", "#f87171", "#94a3b8", "#e879f9"]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

st.set_page_config(page_title="Crime Hotspot & Dispatch", page_icon="🛰️", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.4rem;}
    div[data-testid="stMetric"] {
        background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 14px 16px;
    }
    .hs-label {
        background: rgba(15,23,42,0.9); color: #fff; font: 600 12px/1 system-ui, sans-serif;
        padding: 5px 8px; border-radius: 6px; white-space: nowrap; border: 1px solid rgba(255,255,255,0.4);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def read_upload(data: bytes, name: str) -> pd.DataFrame:
    buf = io.BytesIO(data)
    return pd.read_excel(buf) if name.lower().endswith(".xlsx") else pd.read_csv(buf)


def risk_color(score: float) -> str:
    if score >= 70:
        return "#dc2626"
    if score >= 40:
        return "#f97316"
    return "#eab308"


def build_map(points: pd.DataFrame, hs: pd.DataFrame) -> folium.Map:
    m = folium.Map(
        location=[points.lat.mean(), points.lon.mean()],
        zoom_start=13,
        tiles=None,
        control_scale=True,
    )
    folium.TileLayer(ESRI_IMAGERY, name="Satellite", attr=ESRI_ATTR, max_zoom=19).add_to(m)
    folium.TileLayer(ESRI_LABELS, name="Place names", attr=ESRI_ATTR, overlay=True, max_zoom=19).add_to(m)

    # Individual incidents, coloured by crime type
    types = sorted(points.crime_type.unique())
    colour = {t: PALETTE[i % len(PALETTE)] for i, t in enumerate(types)}
    fg_points = folium.FeatureGroup(name="Incidents", show=False)
    sample = points.sample(min(len(points), 4000), random_state=0)
    for r in sample.itertuples():
        folium.CircleMarker(
            [r.lat, r.lon],
            radius=3,
            color=colour[r.crime_type],
            fill=True,
            fill_color=colour[r.crime_type],
            fill_opacity=0.85,
            weight=0.5,
            tooltip=f"{r.crime_type.title()} · {r.ts:%d %b %Y %H:%M}",
        ).add_to(fg_points)
    fg_points.add_to(m)

    # Density heatmap, weighted by severity and recency
    fg_heat = folium.FeatureGroup(name="Density heatmap", show=False)
    HeatMap(points[["lat", "lon", "weight"]].values.tolist(), radius=18, blur=22).add_to(fg_heat)
    fg_heat.add_to(m)

    # Hotspots with risk colour, radius, and unit assignments
    fg_hs = folium.FeatureGroup(name="Hotspots & dispatch", show=True)
    for r in hs.itertuples():
        colour_hs = risk_color(r.risk_score)
        popup = folium.Popup(
            f"""
            <div style="font:13px system-ui,sans-serif;min-width:230px">
              <b>{r.hotspot_id}</b> · Risk {r.risk_score:.0f}/100<br>
              Area: {r.top_area}<br>
              Incidents: {r.incidents} (last 30 days: {r.last_30d})<br>
              Top crime: {r.top_crime.title()}<br>
              Peak window: {r.peak_window}<br>
              <b>Units assigned: {r.units}</b>
              <div style="color:#64748b;margin-top:4px">{r.lat:.5f}, {r.lon:.5f}</div>
            </div>
            """,
            max_width=320,
        )
        folium.Circle(
            [r.lat, r.lon],
            radius=r.radius_m,
            color=colour_hs,
            weight=2,
            fill=True,
            fill_color=colour_hs,
            fill_opacity=0.2,
        ).add_to(fg_hs)
        folium.CircleMarker(
            [r.lat, r.lon],
            radius=7,
            color="#ffffff",
            weight=2,
            fill=True,
            fill_color=colour_hs,
            fill_opacity=1,
            popup=popup,
        ).add_to(fg_hs)
        folium.Marker(
            [r.lat, r.lon],
            icon=folium.DivIcon(
                html=f'<div class="hs-label">{r.hotspot_id} · {r.units} unit(s)</div>',
                icon_size=(140, 24),
                icon_anchor=(-10, 12),
            ),
        ).add_to(fg_hs)
    fg_hs.add_to(m)

    m.fit_bounds([[points.lat.min(), points.lon.min()], [points.lat.max(), points.lon.max()]])
    folium.LayerControl(collapsed=False).add_to(m)
    Fullscreen().add_to(m)
    return m


def main():
    with st.sidebar:
        st.markdown("### Data")
        upload = st.file_uploader("Incident dataset", type=["csv", "xlsx"])

    st.title("Crime Hotspot & Dispatch Console")
    st.caption("Detects incident hotspots from uploaded records and recommends patrol allocation by time window.")

    if upload is None:
        st.info(
            "Upload a CSV or XLSX of incidents to begin. Required columns: latitude, longitude, and a timestamp. "
            "Crime type and area are optional but improve the output."
        )
        st.stop()

    raw = read_upload(upload.getvalue(), upload.name)
    detected = detect_columns(raw)
    options = ["(none)"] + list(raw.columns)

    with st.sidebar.expander("Column mapping", expanded=not {"lat", "lon", "time"} <= detected.keys()):
        mapping = {}
        labels = [
            ("lat", "Latitude *"),
            ("lon", "Longitude *"),
            ("time", "Timestamp *"),
            ("type", "Crime type"),
            ("area", "Area / station"),
        ]
        for key, label in labels:
            default = detected.get(key, "(none)")
            choice = st.selectbox(label, options, index=options.index(default), key=f"map_{key}")
            if choice != "(none)":
                mapping[key] = choice

    missing = [k for k in ("lat", "lon", "time") if k not in mapping]
    if missing:
        st.error("Map the required columns (latitude, longitude, timestamp) in the sidebar to continue.")
        st.stop()

    df, quality = clean_incidents(raw, mapping)
    if df.empty:
        st.error("No valid records after cleaning. Check the coordinate and timestamp columns.")
        st.stop()

    with st.sidebar:
        st.markdown("### Filters")
        d_min, d_max = df.ts.dt.date.min(), df.ts.dt.date.max()
        date_sel = st.date_input("Date range", value=(d_min, d_max), min_value=d_min, max_value=d_max)
        all_types = sorted(df.crime_type.unique())
        sel_types = st.multiselect("Crime types", all_types, default=all_types)

        st.markdown("### Model")
        min_cluster = st.slider("Minimum hotspot size (incidents)", 5, 100, 15)
        min_samples = st.slider("Core density (min samples)", 1, 30, 5)
        half_life = st.slider("Recency half-life (days)", 30, 730, 180, step=30)

        st.markdown("### Dispatch")
        start_h, end_h = st.select_slider(
            "Time window",
            options=list(range(24)),
            value=(0, 23),
            format_func=lambda h: f"{h:02d}:00",
        )
        units = st.number_input("Patrol units available", min_value=0, max_value=200, value=12)

        st.divider()
        st.caption(
            "Hotspots show where incidents were reported, which reflects both crime and policing patterns. "
            "Use outputs for planning and review, validate with station records, and never as grounds for "
            "action against an individual."
        )

    if isinstance(date_sel, tuple) and len(date_sel) == 2:
        start_date, end_date = date_sel
    else:
        start_date = end_date = date_sel[0] if isinstance(date_sel, tuple) else date_sel

    mask = (
        (df.ts >= pd.Timestamp(start_date))
        & (df.ts < pd.Timestamp(end_date) + pd.Timedelta(days=1))
        & df.crime_type.isin(sel_types)
    )
    view = df[mask].copy()
    if len(view) < min_cluster:
        st.warning(f"Only {len(view)} incidents match the filters. Widen the date range or lower the hotspot size.")
        st.stop()

    with st.spinner("Detecting hotspots..."):
        hs_raw, labeled = detect_hotspots(view, min_cluster, min_samples, half_life)
    if hs_raw.empty:
        st.warning("No hotspots found. Try lowering the minimum hotspot size or core density.")
        st.stop()

    hs = apply_time_window(labeled, hs_raw, start_h, end_h)
    hs = allocate_units(hs, int(units), "window_score")
    hs = hs.sort_values(["window_score", "risk_score"], ascending=False).reset_index(drop=True)

    clustered_share = (labeled.cluster >= 0).mean()
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Incidents analysed", f"{len(labeled):,}")
    k2.metric("Hotspots detected", len(hs))
    k3.metric("Incidents inside hotspots", f"{clustered_share:.0%}")
    k4.metric("Units deployed", f"{int(hs.units.sum())} / {int(units)}")
    top = hs.iloc[0]
    k5.metric("Top hotspot", f"{top.hotspot_id} · {top.top_area}")

    tab_map, tab_dispatch, tab_analytics, tab_quality = st.tabs(
        ["Hotspot map", "Dispatch plan", "Analytics", "Data quality"]
    )

    with tab_map:
        st_folium(build_map(labeled, hs), height=640, use_container_width=True, returned_objects=[])

    with tab_dispatch:
        plan = hs[[
            "hotspot_id", "top_area", "lat", "lon", "radius_m", "risk_score", "window_score",
            "window_incidents", "peak_window", "top_crime", "units",
        ]].rename(columns={
            "hotspot_id": "Hotspot", "top_area": "Area", "lat": "Latitude", "lon": "Longitude",
            "radius_m": "Radius (m)", "risk_score": "Risk", "window_score": "Window score",
            "window_incidents": "Incidents in window", "peak_window": "Peak window",
            "top_crime": "Top crime", "units": "Units",
        })
        st.dataframe(
            plan,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Risk": st.column_config.ProgressColumn("Risk", min_value=0, max_value=100, format="%.0f"),
                "Window score": st.column_config.ProgressColumn("Window score", min_value=0, max_value=100, format="%.0f"),
                "Latitude": st.column_config.NumberColumn(format="%.5f"),
                "Longitude": st.column_config.NumberColumn(format="%.5f"),
            },
        )

        st.subheader("Dispatch orders")
        orders = hs[hs.units > 0]
        if orders.empty:
            st.info("No units assigned. Increase the units available in the sidebar.")
        for _, r in orders.iterrows():
            st.markdown(
                f"**{r.hotspot_id}** ({r.top_area}): send **{int(r.units)}** unit(s) to "
                f"{r.lat:.5f}, {r.lon:.5f} for the {r.peak_window} window, focusing on {r.top_crime}."
            )

        st.download_button(
            "Download dispatch plan (CSV)",
            plan.to_csv(index=False).encode("utf-8"),
            file_name="dispatch_plan.csv",
            mime="text/csv",
        )

    with tab_analytics:
        c1, c2 = st.columns(2)
        by_type = labeled.crime_type.value_counts().reset_index()
        by_type.columns = ["crime_type", "incidents"]
        c1.plotly_chart(
            px.bar(by_type, x="incidents", y="crime_type", orientation="h", title="Incidents by crime type")
            .update_layout(yaxis_title=None, margin=dict(t=50, l=10, r=10, b=10)),
            use_container_width=True,
        )

        pivot = labeled.pivot_table(index="weekday", columns="hour", values="severity", aggfunc="count", fill_value=0)
        pivot = pivot.reindex(WEEKDAYS).fillna(0)
        c2.plotly_chart(
            px.imshow(
                pivot,
                aspect="auto",
                color_continuous_scale="YlOrRd",
                labels=dict(x="Hour", y="", color="Incidents"),
                title="Incidents by weekday and hour",
            ).update_layout(margin=dict(t=50, l=10, r=10, b=10)),
            use_container_width=True,
        )

        monthly = labeled.groupby("month").size().reset_index(name="incidents")
        st.plotly_chart(
            px.line(monthly, x="month", y="incidents", markers=True, title="Monthly incident trend")
            .update_layout(margin=dict(t=50, l=10, r=10, b=10)),
            use_container_width=True,
        )

    with tab_quality:
        st.subheader("Cleaning report")
        q_labels = [
            ("rows_in", "Rows received"),
            ("rows_kept", "Rows used"),
            ("missing_or_invalid_coords", "Invalid coordinates"),
            ("outside_study_area", "Outside study area"),
            ("bad_timestamp", "Unreadable timestamps"),
            ("duplicates_removed", "Duplicates removed"),
        ]
        for row_start in range(0, len(q_labels), 3):
            cols = st.columns(3)
            for col, (key, label) in zip(cols, q_labels[row_start:row_start + 3]):
                col.metric(label, f"{quality[key]:,}")
        st.caption(
            f"Date coverage: {labeled.ts.min():%d %b %Y} to {labeled.ts.max():%d %b %Y}. "
            "Risk weights combine crime severity with recency."
        )


main()
