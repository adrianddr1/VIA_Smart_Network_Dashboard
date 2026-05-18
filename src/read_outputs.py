# -*- coding: utf-8 -*-
"""
Created on Thu May 14 23:31:22 2026

@author: ZhaoJ
"""
from pathlib import Path
import math
import gc

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
# PATHS
# App is in repo/src/read_outputs.py
# Parquet files are in repo/processed/
# =====================================================
DATA_DIR = Path(__file__).resolve().parents[1] / "inputs"

parquet_files = sorted(DATA_DIR.glob("*.parquet"))

if not parquet_files:
    st.error(f"No parquet files found in: {DATA_DIR.resolve()}")
    st.stop()


# =====================================================
# SESSION STATE / MAIN MENU
# =====================================================
if "active_view" not in st.session_state:
    st.session_state.active_view = "Main Menu"


def clear_memory_and_go(view_name):
    st.cache_data.clear()
    gc.collect()
    st.session_state.active_view = view_name
    st.rerun()


selected_file = st.sidebar.selectbox(
    "Select input parquet file",
    parquet_files,
    format_func=lambda p: p.name
)

if st.sidebar.button("Clear cache and reload"):
    st.cache_data.clear()
    gc.collect()
    st.rerun()


# =====================================================
# HELPERS
# =====================================================
@st.cache_data(show_spinner=True, max_entries=1)
def load_parquet(path):
    """
    Load only dashboard-needed columns from parquet.
    Keeps memory lower on Streamlit Cloud.
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
        if c.startswith("delay_minutes_")
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


def prepare_data(df_in):
    """
    Not cached on purpose, so Streamlit does not keep another big copy.
    """
    data = df_in.copy()

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
            data["total_delay_min_cn_filtered"] = pd.concat(
                filtered_parts,
                axis=1
            ).sum(axis=1, skipna=True)
        else:
            data["total_delay_min_cn_filtered"] = 0.0

    data["arrival_ddhhmmss"] = data["arrival_seconds"].apply(seconds_to_ddhhmmss)
    data["departure_ddhhmmss"] = data["departure_seconds"].apply(seconds_to_ddhhmmss)

    for col in ["train_name", "train_type", "train_label", "dp_name"]:
        if col in data.columns:
            data[col] = data[col].astype("category")

    return data, delay_min_cols


def load_and_prepare():
    df_loaded = load_parquet(selected_file)

    if df_loaded.empty:
        st.error("Selected parquet file is empty.")
        st.stop()

    prepared, delay_cols = prepare_data(df_loaded)
    return prepared, delay_cols


def common_filters(df):
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
        "Optional: filter train name(s)",
        train_names,
        default=[],
        help="Leave empty to include all trains."
    )

    if selected_train_names:
        base_df = base_df[base_df["train_name"].astype(str).isin(selected_train_names)]

    if base_df.empty:
        st.warning("No data after filters.")
        st.stop()

    return base_df


# =====================================================
# MAIN MENU
# =====================================================
if st.session_state.active_view == "Main Menu":
    st.subheader("Main Menu")
    st.write("Choose an analysis view. Opening a view clears cached data first to reduce memory issues.")

    c1, c2 = st.columns(2)

    with c1:
        st.button(
            "1. Stringline",
            width="stretch",
            on_click=clear_memory_and_go,
            args=("Stringline",)
        )

        st.button(
            "2. Train Performance Table",
            width="stretch",
            on_click=clear_memory_and_go,
            args=("Train Performance Table",)
        )

    with c2:
        st.button(
            "3. Speed Distribution by Train Name",
            width="stretch",
            on_click=clear_memory_and_go,
            args=("Speed Distribution by Train Name",)
        )

        st.button(
            "4. Average Cumulative Delay by DP",
            width="stretch",
            on_click=clear_memory_and_go,
            args=("Average Cumulative Delay by DP",)
        )

    st.stop()


# =====================================================
# VIEW HEADER + BACK BUTTON
# =====================================================
if st.button("Back to Main Menu / Clear Memory", width="stretch"):
    clear_memory_and_go("Main Menu")

st.divider()

df, delay_min_cols = load_and_prepare()

st.success(f"Loaded: {selected_file.name}")

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
    st.write("Mileage min:", float(df["mileage"].min()))
    st.write("Mileage max:", float(df["mileage"].max()))
    st.write("Arrival hour min:", float(df["arrival_hour"].min()))
    st.write("Arrival hour max:", float(df["arrival_hour"].max()))

base_df = common_filters(df)


# =====================================================
# 1. STRINGLINE
# =====================================================
if st.session_state.active_view == "Stringline":
    st.header("Stringline")

    st.caption(
        "Passenger and freight are differentiated by line color. Each train run is drawn as its own line."
    )

    stringline_df = base_df[
        base_df["mileage"].notna()
        & base_df["arrival_hour"].notna()
    ]

    if stringline_df.empty:
        st.info("No stringline data available.")
        st.stop()

    train_labels = sorted(stringline_df["train_label"].astype(str).dropna().unique())

    selected_train_labels = st.multiselect(
        "Optional: filter specific train runs",
        train_labels,
        default=[],
        help="Leave empty to show all train runs in the 24-hour window."
    )

    if selected_train_labels:
        stringline_df = stringline_df[
            stringline_df["train_label"].astype(str).isin(selected_train_labels)
        ]

    if stringline_df.empty:
        st.info("No train runs selected.")
        st.stop()

    global_min_hour = float(df["arrival_hour"].min())
    global_max_hour = float(df["arrival_hour"].max())

    WINDOW_HOURS = 24.0

    max_start_hour = max(global_min_hour, global_max_hour - WINDOW_HOURS)

    slider_min = int(math.floor(global_min_hour))
    slider_max = int(math.ceil(max_start_hour))

    if slider_max <= slider_min:
        slider_max = slider_min + 1

    start_hour_default = slider_min
    end_hour_default = start_hour_default + WINDOW_HOURS

    chart_df = stringline_df[
        (stringline_df["arrival_hour"] >= start_hour_default)
        & (stringline_df["arrival_hour"] <= end_hour_default)
    ].copy()

    # Placeholder lets the chart appear above the slider while the slider value still controls the chart.
    chart_placeholder = st.empty()

    start_hour = st.slider(
        "Move 24-hour window start time",
        min_value=slider_min,
        max_value=slider_max,
        value=slider_min,
        step=1,
        help="This scrolls the fixed 24-hour window from day 0 through the simulation horizon."
    )

    end_hour = start_hour + WINDOW_HOURS

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Window Start Hour", f"{start_hour:,.0f}")
    c2.metric("Window End Hour", f"{end_hour:,.0f}")
    c3.metric("Window Start Day", f"{start_hour / 24:.1f}")
    c4.metric("Window Size", "24 hrs")

    chart_df = stringline_df[
        (stringline_df["arrival_hour"] >= start_hour)
        & (stringline_df["arrival_hour"] <= end_hour)
    ].copy()

    chart_df = chart_df.sort_values(["train_label", "arrival_hour"])

    st.write(f"Rows in window before chart cap: {len(chart_df):,}")
    st.write(f"Train runs in window: {chart_df['train_label'].nunique():,}")

    if not chart_df.empty:
        st.write(
            f"Mileage range in window: "
            f"{chart_df['mileage'].min():.1f} to {chart_df['mileage'].max():.1f}"
        )

    MAX_CHART_ROWS = 20000

    if len(chart_df) > MAX_CHART_ROWS:
        st.warning(
            f"Too many chart points: {len(chart_df):,}. "
            f"Showing first {MAX_CHART_ROWS:,}. "
            "Use train group, train name, or train run filters if you need a cleaner view."
        )
        chart_df = chart_df.head(MAX_CHART_ROWS)

    if chart_df.empty:
        chart_placeholder.info("No trains in this 24-hour window.")
        st.stop()

    fig = px.line(
        chart_df,
        x="arrival_hour",
        y="mileage",
        color="train_type",
        line_group="train_label",
        markers=False,
        hover_name="train_label",
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
            "train_type": "Train Group",
        },
        title="Stringline by Mileage"
    )

    full_min_mile = float(df["mileage"].min())
    full_max_mile = float(df["mileage"].max())

    fig.update_yaxes(
        range=[full_min_mile, full_max_mile],
        title="Mileage"
    )

    fig.update_xaxes(
        range=[start_hour, end_hour],
        dtick=2,
        title="Simulation Time (hours)"
    )

    fig.update_layout(
        height=760,
        hovermode="closest",
        legend_title_text="Train Group",
    )

    chart_placeholder.plotly_chart(fig, width="stretch")

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
            width="stretch"
        )


# =====================================================
# 2. TRAIN PERFORMANCE TABLE
# =====================================================
elif st.session_state.active_view == "Train Performance Table":
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
            dropna=False,
            observed=False
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

    run_summary = run_summary.replace([float("inf"), -float("inf")], pd.NA)

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
elif st.session_state.active_view == "Speed Distribution by Train Name":
    st.header("Speed Distribution by Train Name")

    run_summary = (
        base_df.groupby(
            ["train_label", "generated_train_id", "train_name", "train_type"],
            dropna=False,
            observed=False
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
        run_summary.groupby(["train_name", "train_type"], dropna=False, observed=False)["avg_speed_mph"]
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
        run_summary.groupby(["train_name", "train_type"], dropna=False, observed=False)
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
        "Optional: filter train names in boxplot",
        available_names,
        default=[],
        help="Leave empty to show all train names."
    )

    if plot_names:
        box_df = run_summary[run_summary["train_name"].astype(str).isin(plot_names)]
    else:
        box_df = run_summary

    if box_df.empty:
        st.info("No data for selected train names.")
        st.stop()

    if len(box_df) > 50000:
        st.warning("Boxplot has too many records. Showing first 50,000.")
        box_df = box_df.head(50000)

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

    st.plotly_chart(fig, width="stretch")


# =====================================================
# 4. AVERAGE CUMULATIVE DELAY BY DP
# =====================================================
elif st.session_state.active_view == "Average Cumulative Delay by DP":
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
        "Optional: filter train names in cumulative delay plot",
        available_delay_names,
        default=[],
        help="Leave empty to show all train names."
    )

    if selected_delay_names:
        plot_delay = avg_delay[
            avg_delay["train_name"].astype(str).isin(selected_delay_names)
        ].copy()
    else:
        plot_delay = avg_delay.copy()

    st.subheader("Average Cumulative Delay Plot")

    if plot_delay.empty:
        st.info("No delay data for selected train names.")
        st.stop()

    if len(plot_delay) > 50000:
        st.warning("Plot has too many records. Showing first 50,000.")
        plot_delay = plot_delay.head(50000)

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

    st.plotly_chart(fig, width="stretch")

    st.subheader("Average Cumulative Delay Table")

    show_grid(
        avg_delay,
        key="avg_cumulative_delay_table",
        height=650,
        page_size=100
    )