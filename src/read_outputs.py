# -*- coding: utf-8 -*-
"""
Created on Thu May 14 23:31:22 2026

@author: ZhaoJ
"""
from pathlib import Path
import math

import pandas as pd
import plotly.express as px
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder


# =====================================================
# PAGE SETUP
# =====================================================
st.set_page_config(layout="wide")
st.title("VIA Smart Network Dashboard")


# =====================================================
# FILE INPUT
# App is in repo/src/read_outputs.py
# Parquet files are in repo/processed/
# =====================================================
DATA_DIR = Path(__file__).resolve().parents[1] / "inputs"

parquet_files = sorted(DATA_DIR.glob("*.parquet"))

if not parquet_files:
    st.error(f"No parquet files found in: {DATA_DIR.resolve()}")
    st.stop()

selected_file = st.sidebar.selectbox(
    "Select input parquet file",
    parquet_files,
    format_func=lambda p: p.name
)


# =====================================================
# HELPERS
# =====================================================
@st.cache_data(show_spinner=True)
def load_parquet(path):
    """
    Load only dashboard-needed columns from parquet.
    This helps prevent Streamlit Cloud crashes.
    """
    import pyarrow.parquet as pq

    schema_cols = pq.read_schema(path).names

    keep_cols = [
        "datapoint_id",
        "generated_train_id",
        "parent_train_id",
        "train_name",
        "train_type",
        "train_label",
        "dp_id",
        "dp_name",
        "mileage",
        "link_id",
        "arrival_seconds",
        "departure_seconds",
        "arrival_hour",
        "dwell_seconds",
        "dwell_minutes",
        "total_delay_min_all_codes",
        "total_delay_min_cn_filtered",
    ]

    delay_cols = [
        c for c in schema_cols
        if c.startswith("delay_code_")
        or c.startswith("delay_code_group_")
        or c.startswith("delay_minutes_")
        or c.startswith("included_by_cn_filter_")
    ]

    cols = [c for c in keep_cols if c in schema_cols] + delay_cols

    return pd.read_parquet(path, columns=cols)


def seconds_to_ddhhmmss(seconds):
    if pd.isna(seconds):
        return ""

    seconds = int(seconds)

    d = seconds // 86400
    seconds %= 86400

    h = seconds // 3600
    seconds %= 3600

    m = seconds // 60
    s = seconds % 60

    return f"{d:03}:{h:02}:{m:02}:{s:02}"


def show_grid(data, key, height=650, page_size=100):
    gb = GridOptionsBuilder.from_dataframe(data)

    gb.configure_default_column(
        sortable=True,
        filter=True,
        resizable=True,
        editable=False
    )

    gb.configure_grid_options(
        pagination=True,
        paginationPageSize=page_size,
        enableRangeSelection=True
    )

    AgGrid(
        data,
        gridOptions=gb.build(),
        height=height,
        fit_columns_on_grid_load=False,
        key=key
    )


def normalize_bool_series(s):
    return (
        s.astype(str)
        .str.strip()
        .str.upper()
        .map({"TRUE": True, "FALSE": False, "1": True, "0": False})
        .fillna(False)
    )


@st.cache_data(show_spinner=True)
def prepare_data(df_in):
    data = df_in.copy()

    # -----------------------------
    # Numeric cleanup
    # -----------------------------
    numeric_cols = [
        "datapoint_id",
        "generated_train_id",
        "parent_train_id",
        "dp_id",
        "link_id",
        "mileage",
        "arrival_seconds",
        "departure_seconds",
        "arrival_hour",
        "dwell_seconds",
        "dwell_minutes",
        "total_delay_min_all_codes",
        "total_delay_min_cn_filtered",
    ]

    for col in numeric_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    # -----------------------------
    # Safe string cleanup
    # Fixes Arrow/category/string concat issue
    # -----------------------------
    if "train_name" not in data.columns:
        data["train_name"] = ""

    data["train_name"] = data["train_name"].astype("string").fillna("")

    if "train_type" not in data.columns:
        data["train_type"] = data["train_name"].apply(
            lambda x: "Passenger" if isinstance(x, str) and x.startswith("P") else "Freight / Other"
        )

    data["train_type"] = data["train_type"].astype("string").fillna("Unknown")

    if "dp_name" in data.columns:
        data["dp_name"] = data["dp_name"].astype("string").fillna("")
    else:
        data["dp_name"] = ""

    data["generated_train_id_str"] = data["generated_train_id"].astype("string").fillna("")

    if "train_label" not in data.columns:
        data["train_label"] = data["generated_train_id_str"] + " | " + data["train_name"]
    else:
        data["train_label"] = data["train_label"].astype("string").fillna("")
        missing_label = data["train_label"].str.strip().eq("")
        data.loc[missing_label, "train_label"] = (
            data.loc[missing_label, "generated_train_id_str"]
            + " | "
            + data.loc[missing_label, "train_name"]
        )

    data = data.drop(columns=["generated_train_id_str"])

    # -----------------------------
    # If total delay columns are missing, rebuild them
    # -----------------------------
    delay_min_cols = sorted(
        [c for c in data.columns if c.startswith("delay_minutes_")],
        key=lambda x: int(x.split("_")[-1])
    )

    for col in delay_min_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0)

    if "total_delay_min_all_codes" not in data.columns:
        if delay_min_cols:
            data["total_delay_min_all_codes"] = data[delay_min_cols].sum(axis=1, skipna=True)
        else:
            data["total_delay_min_all_codes"] = 0.0

    if "total_delay_min_cn_filtered" not in data.columns:
        filtered_parts = []

        for delay_col in delay_min_cols:
            seq = delay_col.split("_")[-1]
            include_col = f"included_by_cn_filter_{seq}"

            if include_col in data.columns:
                include_mask = normalize_bool_series(data[include_col])
                filtered_parts.append(data[delay_col].where(include_mask, 0))
            else:
                filtered_parts.append(data[delay_col])

        if filtered_parts:
            data["total_delay_min_cn_filtered"] = pd.concat(filtered_parts, axis=1).sum(axis=1, skipna=True)
        else:
            data["total_delay_min_cn_filtered"] = 0.0

    # -----------------------------
    # Display time
    # -----------------------------
    data["arrival_ddhhmmss"] = data["arrival_seconds"].apply(seconds_to_ddhhmmss)
    data["departure_ddhhmmss"] = data["departure_seconds"].apply(seconds_to_ddhhmmss)

    # -----------------------------
    # Reduce display memory
    # -----------------------------
    for col in ["train_name", "train_type", "train_label", "dp_name"]:
        if col in data.columns:
            data[col] = data[col].astype("category")

    return data, delay_min_cols


# =====================================================
# LOAD DATA
# =====================================================
df = load_parquet(selected_file)

if df.empty:
    st.error("Selected parquet file is empty.")
    st.stop()

df, delay_min_cols = prepare_data(df)

st.success(f"Loaded: {selected_file.name}")


# =====================================================
# TOP METRICS
# =====================================================
c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("Rows", f"{len(df):,}")
c2.metric("Train Runs", f"{df['train_label'].nunique():,}")
c3.metric("Train Names", f"{df['train_name'].nunique():,}")
c4.metric("Decision Points", f"{df['dp_id'].nunique():,}")
c5.metric("Delay Columns", len(delay_min_cols))

with st.expander("Debug / file info"):
    st.write("Data folder:", DATA_DIR.resolve())
    st.write("Loaded file:", selected_file.name)
    st.write("Columns loaded:", list(df.columns))
    st.write("Memory MB:", round(df.memory_usage(deep=True).sum() / 1024 / 1024, 1))


# =====================================================
# VIEW SELECTOR
# =====================================================
view = st.sidebar.radio(
    "Select view",
    [
        "1. Stringline",
        "2. Train Performance Table",
        "3. Speed Distribution by Train Name",
        "4. Average Cumulative Delay by DP",
    ]
)


# =====================================================
# COMMON FILTERS
# =====================================================
st.sidebar.header("Common Filters")

train_type_filter = st.sidebar.selectbox(
    "Train group",
    ["All", "Passenger", "Freight / Other", "Unknown"]
)

base_df = df

if train_type_filter != "All":
    base_df = base_df[base_df["train_type"].astype(str) == train_type_filter]

train_names = sorted(base_df["train_name"].astype(str).dropna().unique())

selected_train_names = st.sidebar.multiselect(
    "Select train name(s)",
    train_names,
    default=train_names[:5]
)

if selected_train_names:
    base_df = base_df[base_df["train_name"].astype(str).isin(selected_train_names)]

if base_df.empty:
    st.warning("No data after filters.")
    st.stop()


# =====================================================
# 1. STRINGLINE
# =====================================================
if view == "1. Stringline":
    st.header("Stringline")

    st.caption(
        "This view uses a fixed 24-hour window. Move the window start slider to scroll through the simulation."
    )

    train_labels = sorted(base_df["train_label"].astype(str).dropna().unique())

    selected_train_labels = st.multiselect(
        "Select specific train runs",
        train_labels,
        default=train_labels[:3]
    )

    stringline_df = base_df

    if selected_train_labels:
        stringline_df = stringline_df[
            stringline_df["train_label"].astype(str).isin(selected_train_labels)
        ]

    if stringline_df.empty:
        st.info("No train runs selected.")
        st.stop()

    min_hour = float(stringline_df["arrival_hour"].min())
    max_hour = float(stringline_df["arrival_hour"].max())

    # Fixed 24-hour window
    WINDOW_HOURS = 24.0

    # Use full data max, not selected max, so slider can go from day 0 to full horizon.
    global_min_hour = float(df["arrival_hour"].min())
    global_max_hour = float(df["arrival_hour"].max())

    max_start_hour = max(global_min_hour, global_max_hour - WINDOW_HOURS)

    # Keep integer hour slider for stability on Streamlit Cloud
    slider_min = int(math.floor(global_min_hour))
    slider_max = int(math.ceil(max_start_hour))

    if slider_max <= slider_min:
        slider_max = slider_min + 1

    start_hour = st.slider(
        "Move 24-hour window start time",
        min_value=slider_min,
        max_value=slider_max,
        value=slider_min,
        step=1,
        help="Move this to scroll through the simulation. Window is always 24 hours."
    )

    end_hour = start_hour + WINDOW_HOURS

    c1, c2, c3 = st.columns(3)
    c1.metric("Window Start", seconds_to_ddhhmmss(start_hour * 3600))
    c2.metric("Window End", seconds_to_ddhhmmss(end_hour * 3600))
    c3.metric("Window Size", "24 hours")

    chart_df = stringline_df[
        (stringline_df["arrival_hour"] >= start_hour)
        & (stringline_df["arrival_hour"] <= end_hour)
        & (stringline_df["mileage"].notna())
    ].copy()

    chart_df = chart_df.sort_values(["train_label", "arrival_hour"])

    st.write(f"Stringline rows in 24-hour window: {len(chart_df):,}")

    MAX_CHART_ROWS = 8000

    if len(chart_df) > MAX_CHART_ROWS:
        st.warning(
            f"Too many chart points: {len(chart_df):,}. "
            f"Showing first {MAX_CHART_ROWS:,}. Select fewer train names or train runs."
        )
        chart_df = chart_df.head(MAX_CHART_ROWS)

    if chart_df.empty:
        st.info("No data in this 24-hour window.")
        st.stop()

    fig = px.line(
        chart_df,
        x="arrival_hour",
        y="mileage",
        color="train_label",
        markers=False,
        hover_data=[
            "train_name",
            "train_type",
            "dp_name",
            "dp_id",
            "mileage",
            "arrival_ddhhmmss",
            "departure_ddhhmmss",
            "dwell_minutes",
            "total_delay_min_all_codes",
            "total_delay_min_cn_filtered",
        ],
        labels={
            "arrival_hour": "Simulation Time (hours)",
            "mileage": "Mileage",
            "train_label": "Train Run",
        },
        title="Stringline by Mileage"
    )

    # Fixed visible 24-hour window
    fig.update_xaxes(
        range=[start_hour, end_hour],
        dtick=2,
    )

    # Zoom/pan still available from Plotly toolbar
    fig.update_layout(
        height=760,
        hovermode="closest",
        legend_title_text="Train Run",
    )

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Filtered stringline data"):
        st.dataframe(
            chart_df[
                [
                    "train_label",
                    "train_name",
                    "train_type",
                    "dp_name",
                    "dp_id",
                    "mileage",
                    "arrival_ddhhmmss",
                    "departure_ddhhmmss",
                    "dwell_minutes",
                    "total_delay_min_all_codes",
                    "total_delay_min_cn_filtered",
                ]
            ].head(3000),
            use_container_width=True
        )


# =====================================================
# 2. TRAIN PERFORMANCE TABLE
# =====================================================
elif view == "2. Train Performance Table":
    st.header("Train Performance Table")

    delay_basis = st.radio(
        "Delay basis",
        ["All delay codes", "CN-filtered delay only"],
        horizontal=True
    )

    delay_col = (
        "total_delay_min_all_codes"
        if delay_basis == "All delay codes"
        else "total_delay_min_cn_filtered"
    )

    run_summary = (
        base_df.groupby(
            ["train_label", "generated_train_id", "train_name", "train_type"],
            dropna=False
        )
        .agg(
            first_arrival_seconds=("arrival_seconds", "min"),
            last_departure_seconds=("departure_seconds", "max"),
            min_mileage=("mileage", "min"),
            max_mileage=("mileage", "max"),
            datapoints=("datapoint_id", "count"),
            total_dwell_min=("dwell_minutes", "sum"),
            max_dwell_min=("dwell_minutes", "max"),
            total_delay_min=(delay_col, "sum"),
            max_delay_min=(delay_col, "max"),
        )
        .reset_index()
    )

    run_summary["trip_duration_hr"] = (
        run_summary["last_departure_seconds"] - run_summary["first_arrival_seconds"]
    ) / 3600

    run_summary["route_miles"] = (
        run_summary["max_mileage"] - run_summary["min_mileage"]
    ).abs()

    run_summary["avg_speed_mph"] = (
        run_summary["route_miles"] / run_summary["trip_duration_hr"]
    )

    run_summary["first_arrival"] = run_summary["first_arrival_seconds"].apply(seconds_to_ddhhmmss)
    run_summary["last_departure"] = run_summary["last_departure_seconds"].apply(seconds_to_ddhhmmss)

    run_summary = run_summary[
        [
            "train_label",
            "generated_train_id",
            "train_name",
            "train_type",
            "first_arrival",
            "last_departure",
            "route_miles",
            "trip_duration_hr",
            "avg_speed_mph",
            "datapoints",
            "total_dwell_min",
            "max_dwell_min",
            "total_delay_min",
            "max_delay_min",
        ]
    ].sort_values("avg_speed_mph")

    show_grid(
        run_summary,
        key="train_performance_table",
        height=700,
        page_size=100
    )


# =====================================================
# 3. SPEED DISTRIBUTION BY TRAIN NAME
# =====================================================
elif view == "3. Speed Distribution by Train Name":
    st.header("Speed Distribution by Train Name")

    run_summary = (
        base_df.groupby(
            ["train_label", "generated_train_id", "train_name", "train_type"],
            dropna=False
        )
        .agg(
            first_arrival_seconds=("arrival_seconds", "min"),
            last_departure_seconds=("departure_seconds", "max"),
            min_mileage=("mileage", "min"),
            max_mileage=("mileage", "max"),
        )
        .reset_index()
    )

    run_summary["trip_duration_hr"] = (
        run_summary["last_departure_seconds"] - run_summary["first_arrival_seconds"]
    ) / 3600

    run_summary["route_miles"] = (
        run_summary["max_mileage"] - run_summary["min_mileage"]
    ).abs()

    run_summary["avg_speed_mph"] = (
        run_summary["route_miles"] / run_summary["trip_duration_hr"]
    )

    run_summary = run_summary.replace([float("inf"), -float("inf")], pd.NA)
    run_summary = run_summary[run_summary["avg_speed_mph"].notna()]

    speed_stats = (
        run_summary.groupby(["train_name", "train_type"], dropna=False)["avg_speed_mph"]
        .quantile([0, 0.25, 0.50, 0.75, 1.0])
        .unstack()
        .reset_index()
        .rename(
            columns={
                0: "min_mph",
                0.25: "p25_mph",
                0.5: "p50_mph",
                0.75: "p75_mph",
                1.0: "max_mph",
            }
        )
    )

    run_counts = (
        run_summary.groupby(["train_name", "train_type"], dropna=False)
        .size()
        .reset_index(name="train_runs")
    )

    speed_stats = speed_stats.merge(
        run_counts,
        on=["train_name", "train_type"],
        how="left"
    )

    st.subheader("Speed Percentile Table")
    show_grid(
        speed_stats.sort_values("p50_mph"),
        key="speed_stats_table",
        height=500,
        page_size=100
    )

    st.subheader("Speed Boxplot")

    available_names = sorted(run_summary["train_name"].astype(str).dropna().unique())

    plot_names = st.multiselect(
        "Train names to include in boxplot",
        available_names,
        default=available_names[:20]
    )

    box_df = run_summary[run_summary["train_name"].astype(str).isin(plot_names)]

    if box_df.empty:
        st.info("No data for selected train names.")
        st.stop()

    fig = px.box(
        box_df,
        x="train_name",
        y="avg_speed_mph",
        color="train_type",
        points="outliers",
        title="Average Speed Distribution by Train Name",
        labels={
            "train_name": "Train Name",
            "avg_speed_mph": "Average Speed (mph)",
        },
    )

    fig.update_layout(
        height=700,
        xaxis_tickangle=-45
    )

    st.plotly_chart(fig, use_container_width=True)


# =====================================================
# 4. AVERAGE CUMULATIVE DELAY BY DP
# =====================================================
elif view == "4. Average Cumulative Delay by DP":
    st.header("Average Cumulative Delay by DP")

    delay_basis = st.radio(
        "Delay basis",
        ["All delay codes", "Exclude CN-excluded delay codes"],
        horizontal=True
    )

    delay_col = (
        "total_delay_min_all_codes"
        if delay_basis == "All delay codes"
        else "total_delay_min_cn_filtered"
    )

    delay_df = base_df[
        [
            "train_label",
            "train_name",
            "train_type",
            "generated_train_id",
            "dp_id",
            "dp_name",
            "mileage",
            "arrival_seconds",
            delay_col,
        ]
    ].copy()

    delay_df = delay_df.sort_values(["train_label", "arrival_seconds"])

    delay_df["cumulative_delay_min"] = (
        delay_df.groupby("train_label", observed=False)[delay_col].cumsum()
    )

    avg_delay = (
        delay_df.groupby(
            ["train_name", "train_type", "dp_id", "dp_name", "mileage"],
            dropna=False,
            observed=False
        )
        .agg(
            avg_cumulative_delay_min=("cumulative_delay_min", "mean"),
            p50_cumulative_delay_min=("cumulative_delay_min", "median"),
            max_cumulative_delay_min=("cumulative_delay_min", "max"),
            train_runs=("train_label", "nunique"),
        )
        .reset_index()
        .sort_values(["train_name", "mileage"])
    )

    available_delay_names = sorted(avg_delay["train_name"].astype(str).dropna().unique())

    selected_delay_names = st.multiselect(
        "Select train names for cumulative delay plot",
        available_delay_names,
        default=available_delay_names[:10]
    )

    plot_delay = avg_delay[
        avg_delay["train_name"].astype(str).isin(selected_delay_names)
    ].copy()

    st.subheader("Average Cumulative Delay Plot")

    if plot_delay.empty:
        st.info("No delay data for selected train names.")
        st.stop()

    fig = px.line(
        plot_delay,
        x="mileage",
        y="avg_cumulative_delay_min",
        color="train_name",
        markers=True,
        hover_data=[
            "train_type",
            "dp_id",
            "dp_name",
            "mileage",
            "avg_cumulative_delay_min",
            "p50_cumulative_delay_min",
            "max_cumulative_delay_min",
            "train_runs",
        ],
        title="Average Cumulative Delay by DP",
        labels={
            "mileage": "Mileage",
            "avg_cumulative_delay_min": "Average Cumulative Delay (min)",
            "train_name": "Train Name",
        },
    )

    fig.update_layout(
        height=720,
        hovermode="closest"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Average Cumulative Delay Table")

    show_grid(
        avg_delay,
        key="avg_cumulative_delay_table",
        height=650,
        page_size=100
    )