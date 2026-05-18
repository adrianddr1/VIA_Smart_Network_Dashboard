# -*- coding: utf-8 -*-
"""
Created on Thu May 14 23:31:22 2026

@author: ZhaoJ
"""
from pathlib import Path
import re

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


@st.cache_data(show_spinner=True)
def load_parquet(path):
    # Only load columns needed by dashboard
    keep_cols = [
        "datapoint_id",
        "generated_train_id",
        "dp_id",
        "link_id",
        "arrival_time",
        "departure_time",
        "arrival_seconds",
        "departure_seconds",
        "dwell_minutes",
        "arrival_hour",
        "parent_train_id",
        "train_name",
        "train_type",
        "dp_name",
        "mileage",
        "train_label",
    ]

    # Add delay columns if present
    import pyarrow.parquet as pq
    schema_cols = pq.read_schema(path).names

    delay_cols = [
        c for c in schema_cols
        if c.startswith("delay_minutes_")
        or c.startswith("included_by_cn_filter_")
    ]

    cols = [c for c in keep_cols if c in schema_cols] + delay_cols

    return pd.read_parquet(path, columns=cols)



# =====================================================
# HELPER FUNCTIONS
# =====================================================
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


def ddhhmmss_to_seconds(text):
    if not isinstance(text, str):
        return None

    m = re.match(r"^(\d+):(\d{1,2}):(\d{1,2}):(\d{1,2})$", text.strip())

    if not m:
        return None

    d, h, mnt, s = map(int, m.groups())
    return d * 86400 + h * 3600 + mnt * 60 + s


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
    """
    Handles bool, TRUE/FALSE strings, 1/0, and missing values.
    """
    return (
        s.astype(str)
        .str.strip()
        .str.upper()
        .map({"TRUE": True, "FALSE": False, "1": True, "0": False})
        .fillna(False)
    )


def add_delay_columns(data):
    """
    Creates:
    - row_delay_all_min
    - row_delay_cn_filtered_min

    Uses delay_minutes_1, delay_minutes_2, ...
    and included_by_cn_filter_1, included_by_cn_filter_2, ...
    """

    data = data.copy()

    delay_min_cols = sorted(
        [c for c in data.columns if c.startswith("delay_minutes_")],
        key=lambda x: int(x.split("_")[-1])
    )

    data[delay_min_cols] = data[delay_min_cols].apply(
        pd.to_numeric,
        errors="coerce"
    )

    if not delay_min_cols:
        data["row_delay_all_min"] = 0.0
        data["row_delay_cn_filtered_min"] = 0.0
        return data, delay_min_cols

    data["row_delay_all_min"] = data[delay_min_cols].sum(axis=1, skipna=True)

    filtered_delay_parts = []

    for delay_col in delay_min_cols:
        seq = delay_col.split("_")[-1]
        filter_col = f"included_by_cn_filter_{seq}"

        if filter_col in data.columns:
            include_mask = normalize_bool_series(data[filter_col])
            filtered_delay_parts.append(data[delay_col].where(include_mask, 0))
        else:
            filtered_delay_parts.append(data[delay_col].fillna(0))

    if filtered_delay_parts:
        data["row_delay_cn_filtered_min"] = pd.concat(
            filtered_delay_parts,
            axis=1
        ).sum(axis=1, skipna=True)
    else:
        data["row_delay_cn_filtered_min"] = 0.0

    return data, delay_min_cols


@st.cache_data(show_spinner=True)
def prepare_data(df_in):
    data = df_in.copy()

    data["mileage"] = pd.to_numeric(data["mileage"], errors="coerce")
    data["arrival_seconds"] = pd.to_numeric(data["arrival_seconds"], errors="coerce")
    data["departure_seconds"] = pd.to_numeric(data["departure_seconds"], errors="coerce")
    data["dwell_minutes"] = pd.to_numeric(data["dwell_minutes"], errors="coerce").fillna(0)

    data["arrival_hour"] = data["arrival_seconds"] / 3600
    data["departure_hour"] = data["departure_seconds"] / 3600

    data["arrival_ddhhmmss"] = data["arrival_seconds"].apply(seconds_to_ddhhmmss)
    data["departure_ddhhmmss"] = data["departure_seconds"].apply(seconds_to_ddhhmmss)

    data["train_name"] = data["train_name"].fillna("")
    data["train_label"] = data["train_label"].fillna(
        data["generated_train_id"].astype(str) + " | " + data["train_name"]
    )

    if "train_type" not in data.columns:
        data["train_type"] = data["train_name"].apply(
            lambda x: "Passenger" if isinstance(x, str) and x.startswith("P") else "Freight / Other"
        )

    data, delay_min_cols = add_delay_columns(data)

    return data, delay_min_cols


df, delay_min_cols = prepare_data(df)


# =====================================================
# METRICS
# =====================================================
c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("Rows", f"{len(df):,}")
c2.metric("Train Runs", f"{df['train_label'].nunique():,}")
c3.metric("Train Names", f"{df['train_name'].nunique():,}")
c4.metric("Decision Points", f"{df['dp_id'].nunique():,}")
c5.metric("Delay Columns", len(delay_min_cols))


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
    ["All", "Passenger", "Freight / Other"]
)

base_df = df

if train_type_filter != "All":
    base_df = base_df[base_df["train_type"] == train_type_filter]

train_names = sorted(base_df["train_name"].dropna().unique())

selected_train_names = st.sidebar.multiselect(
    "Select train name(s)",
    train_names,
    default=train_names[:10]
)

if selected_train_names:
    base_df = base_df[base_df["train_name"].isin(selected_train_names)]

if base_df.empty:
    st.warning("No data after filters.")
    st.stop()


# =====================================================
# VIEW 1: STRINGLINE
# =====================================================
if view == "1. Stringline":
    st.header("Stringline")

    train_labels = sorted(base_df["train_label"].dropna().unique())

    selected_train_labels = st.multiselect(
        "Highlight / plot specific train runs",
        train_labels,
        default=train_labels[:10]
    )

    stringline_df = base_df.copy()

    if selected_train_labels:
        stringline_df = stringline_df[stringline_df["train_label"].isin(selected_train_labels)]

    min_sec = stringline_df["arrival_seconds"].min()
    max_sec = stringline_df["arrival_seconds"].max()

    default_start = seconds_to_ddhhmmss(min_sec)
    default_end = seconds_to_ddhhmmss(min(min_sec + 24 * 3600, max_sec))

    c1, c2 = st.columns(2)

    start_text = c1.text_input(
        "Start time DD:HH:MM:SS",
        value=default_start
    )

    end_text = c2.text_input(
        "End time DD:HH:MM:SS",
        value=default_end
    )

    start_sec = ddhhmmss_to_seconds(start_text)
    end_sec = ddhhmmss_to_seconds(end_text)

    if start_sec is None or end_sec is None:
        st.error("Invalid time format. Use DD:HH:MM:SS, for example 000:00:00:00.")
        st.stop()

    stringline_df = stringline_df[
        (stringline_df["arrival_seconds"] >= start_sec)
        & (stringline_df["arrival_seconds"] <= end_sec)
        & (stringline_df["mileage"].notna())
    ].copy()

    stringline_df = stringline_df.sort_values(["train_label", "arrival_seconds"])

    st.write(f"Stringline rows: {len(stringline_df):,}")

    MAX_CHART_ROWS = 3000

    if len(stringline_df) > MAX_CHART_ROWS:
        st.warning(
            f"Too many rows for the chart ({len(stringline_df):,}). "
            f"Showing first {MAX_CHART_ROWS:,}. Select fewer trains or a shorter time window."
        )
        stringline_df = stringline_df.head(MAX_CHART_ROWS)

    if stringline_df.empty:
        st.info("No data in selected window.")
        st.stop()

    fig = px.line(
        stringline_df,
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
            "row_delay_all_min",
            "row_delay_cn_filtered_min",
        ],
        labels={
            "arrival_hour": "Simulation Time (hours)",
            "mileage": "Mileage",
            "train_label": "Train Run",
        },
        title="Stringline by Train Run"
    )

    # Plotly zoom/pan is built in.
    # Range slider underneath allows scrolling through the full selected range.
    fig.update_xaxes(
        range=[start_sec / 3600, end_sec / 3600],
        rangeslider=dict(visible=True),
        rangeselector=dict(
            buttons=[
                dict(count=6, label="6h", step="hour", stepmode="backward"),
                dict(count=12, label="12h", step="hour", stepmode="backward"),
                dict(count=24, label="24h", step="hour", stepmode="backward"),
                dict(step="all", label="All"),
            ]
        )
    )

    fig.update_layout(
        height=760,
        hovermode="closest",
        legend_title_text="Train Run",
    )

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Filtered stringline data"):
        st.dataframe(
            stringline_df[
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
                    "row_delay_all_min",
                    "row_delay_cn_filtered_min",
                ]
            ].head(5000),
            use_container_width=True
        )


# =====================================================
# VIEW 2: TRAIN PERFORMANCE TABLE
# =====================================================
elif view == "2. Train Performance Table":
    st.header("Train Performance Table")

    delay_basis = st.radio(
        "Delay basis",
        ["All delay codes", "CN-filtered delay only"],
        horizontal=True
    )

    delay_col = (
        "row_delay_all_min"
        if delay_basis == "All delay codes"
        else "row_delay_cn_filtered_min"
    )

    run_summary = (
        base_df.groupby(["train_label", "generated_train_id", "train_name", "train_type"], dropna=False)
        .agg(
            first_arrival_seconds=("arrival_seconds", "min"),
            last_departure_seconds=("departure_seconds", "max"),
            first_mileage=("mileage", "first"),
            last_mileage=("mileage", "last"),
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

    # Use mileage range as practical distance covered.
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
# VIEW 3: SPEED DISTRIBUTION BY TRAIN NAME
# =====================================================
elif view == "3. Speed Distribution by Train Name":
    st.header("Speed Distribution by Train Name")

    run_summary = (
        base_df.groupby(["train_label", "generated_train_id", "train_name", "train_type"], dropna=False)
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

    run_summary = run_summary[
        run_summary["avg_speed_mph"].replace([float("inf"), -float("inf")], pd.NA).notna()
    ]

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

    plot_limit_names = st.multiselect(
        "Train names to include in boxplot",
        sorted(run_summary["train_name"].dropna().unique()),
        default=sorted(run_summary["train_name"].dropna().unique())[:20]
    )

    box_df = run_summary[run_summary["train_name"].isin(plot_limit_names)]

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
# VIEW 4: AVERAGE CUMULATIVE DELAY BY DP
# =====================================================
elif view == "4. Average Cumulative Delay by DP":
    st.header("Average Cumulative Delay by DP")

    delay_basis = st.radio(
        "Delay basis",
        ["All delay codes", "Exclude CN-excluded delay codes"],
        horizontal=True
    )

    delay_col = (
        "row_delay_all_min"
        if delay_basis == "All delay codes"
        else "row_delay_cn_filtered_min"
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
        delay_df.groupby("train_label")[delay_col].cumsum()
    )

    # Average repeated daily/synthetic runs by train_name and DP.
    avg_delay = (
        delay_df.groupby(["train_name", "train_type", "dp_id", "dp_name", "mileage"], dropna=False)
        .agg(
            avg_cumulative_delay_min=("cumulative_delay_min", "mean"),
            p50_cumulative_delay_min=("cumulative_delay_min", "median"),
            max_cumulative_delay_min=("cumulative_delay_min", "max"),
            train_runs=("train_label", "nunique"),
        )
        .reset_index()
        .sort_values(["train_name", "mileage"])
    )

    selected_delay_train_names = st.multiselect(
        "Select train names for cumulative delay plot",
        sorted(avg_delay["train_name"].dropna().unique()),
        default=sorted(avg_delay["train_name"].dropna().unique())[:10]
    )

    plot_delay = avg_delay[
        avg_delay["train_name"].isin(selected_delay_train_names)
    ].copy()

    st.subheader("Average Cumulative Delay Plot")

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