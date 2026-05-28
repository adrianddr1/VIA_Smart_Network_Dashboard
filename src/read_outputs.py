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
import plotly.graph_objects as go
import streamlit as st
import yaml
import streamlit_authenticator as stauth
from yaml.loader import SafeLoader


# =====================================================
# PAGE SETUP
# Must be before other Streamlit commands
# =====================================================
st.set_page_config(layout="wide")
st.title("VIA Smart Network Dashboard")


# =====================================================
# AUTHENTICATION
# repo/
# └── src/
#     ├── read_outputs.py
#     └── config.yaml
# =====================================================
CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"

if not CONFIG_PATH.exists():
    st.error(f"Authentication config not found at: {CONFIG_PATH}")
    st.write("Current file:", Path(__file__).resolve())
    st.write("Current folder:", Path(__file__).resolve().parent)
    st.stop()

with open(CONFIG_PATH, "r", encoding="utf-8") as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config["credentials"],
    config["cookie"]["name"],
    config["cookie"]["key"],
    config["cookie"]["expiry_days"],
)

try:
    authenticator.login(location="main")
except Exception as e:
    st.error(e)
    st.stop()

if st.session_state.get("authentication_status") is False:
    st.error("Username or password is incorrect.")
    st.stop()

elif st.session_state.get("authentication_status") is None:
    st.warning("Please enter your username and password.")
    st.stop()

elif st.session_state.get("authentication_status"):
    authenticator.logout("Logout", "sidebar")
    st.sidebar.success(f"Logged in as {st.session_state.get('name')}")


# =====================================================
# PATHS
# repo/
# ├── src/read_outputs.py
# └── inputs/*.parquet
# =====================================================
DATA_DIR = Path(__file__).resolve().parents[1] / "inputs"

parquet_files = sorted(DATA_DIR.glob("*.parquet"))

if not parquet_files:
    st.error(f"No parquet files found in: {DATA_DIR.resolve()}")
    st.stop()


# =====================================================
# SESSION STATE
# =====================================================
if "active_view" not in st.session_state:
    st.session_state.active_view = "Main Menu"


def clear_memory():
    st.cache_data.clear()
    gc.collect()


selected_file = st.sidebar.selectbox(
    "Select input parquet file",
    parquet_files,
    format_func=lambda p: p.name
)

if st.sidebar.button("Clear cache and reload"):
    clear_memory()
    st.rerun()


# =====================================================
# HELPERS
# =====================================================
@st.cache_data(show_spinner=True, max_entries=1)
def load_parquet(path):
    """
    Load only dashboard-needed columns.
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


def normalize_bool_series(s):
    return (
        s.astype(str)
        .str.strip()
        .str.upper()
        .map({"TRUE": True, "FALSE": False, "1": True, "0": False})
        .fillna(False)
    )


def safe_dataframe(df, max_rows=5000):
    st.caption(f"Showing first {min(len(df), max_rows):,} rows out of {len(df):,}.")
    st.dataframe(df.head(max_rows), width="stretch")


def prepare_data(df_in):
    """
    Do not cache this to avoid duplicate cached dataframe copies.
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

    # Flag records with impossible/incomplete dispatch output.
    # These are excluded from all operational analyses.
    data["never_dispatched_record"] = (
        (data["departure_seconds"] == 0)
        & (data["arrival_seconds"] > 0)
    )

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
# TPC HELPERS
# =====================================================
def make_two_point_stringline_rows(df, time_prefix):
    """
    Convert each DP record into two plotting points:
    arrival and departure at the same mileage.
    This makes dwell appear as a vertical segment at the DP location.
    """
    rows = []

    for _, r in df.iterrows():
        rows.append(
            {
                "plot_time_seconds": r["arrival_seconds"] - time_prefix,
                "plot_time_hours": (r["arrival_seconds"] - time_prefix) / 3600,
                "mileage": r["mileage"],
                "dp_id": r["dp_id"],
                "dp_name": r["dp_name"],
                "event": "Arrival",
                "train_label": r.get("train_label", ""),
                "train_name": r.get("train_name", ""),
                "train_type": r.get("train_type", ""),
                "arrival_seconds": r["arrival_seconds"],
                "departure_seconds": r["departure_seconds"],
                "dwell_minutes": r.get("dwell_minutes", 0),
            }
        )

        rows.append(
            {
                "plot_time_seconds": r["departure_seconds"] - time_prefix,
                "plot_time_hours": (r["departure_seconds"] - time_prefix) / 3600,
                "mileage": r["mileage"],
                "dp_id": r["dp_id"],
                "dp_name": r["dp_name"],
                "event": "Departure",
                "train_label": r.get("train_label", ""),
                "train_name": r.get("train_name", ""),
                "train_type": r.get("train_type", ""),
                "arrival_seconds": r["arrival_seconds"],
                "departure_seconds": r["departure_seconds"],
                "dwell_minutes": r.get("dwell_minutes", 0),
            }
        )

    out = pd.DataFrame(rows)

    if not out.empty:
        out = out.sort_values("plot_time_seconds")

    return out


def build_tpc_profile_for_train_name(data, selected_train_name):
    """
    Build TPC/base profile for one train_name.

    Logic:
    1. Pick the most complete train_label as the route template.
    2. For each DP, use minimum non-negative dwell time.
    3. For each consecutive DP pair, use minimum observed link time:
       downstream arrival - upstream departure.
    4. Rebuild a synthetic minimum-time stringline starting at time zero.
    """
    tn = data[
        (data["train_name"].astype(str) == selected_train_name)
        & (data["arrival_seconds"].notna())
        & (data["departure_seconds"].notna())
        & (data["mileage"].notna())
    ].copy()

    if tn.empty:
        return pd.DataFrame(), pd.DataFrame(), None, pd.DataFrame()

    tn = tn[
        (tn["departure_seconds"] >= tn["arrival_seconds"])
        & (tn["departure_seconds"] > 0)
    ].copy()

    if tn.empty:
        return pd.DataFrame(), pd.DataFrame(), None, pd.DataFrame()

    # Pick most complete train run as the route template.
    route_label = (
        tn.groupby("train_label", observed=True)
        .size()
        .sort_values(ascending=False)
        .index[0]
    )

    route = (
        tn[tn["train_label"] == route_label]
        .sort_values("arrival_seconds")
        .copy()
    )

    route = route.drop_duplicates(
        subset=["dp_id", "arrival_seconds", "mileage"],
        keep="first"
    ).reset_index(drop=True)

    if len(route) < 2:
        return pd.DataFrame(), pd.DataFrame(), route_label, pd.DataFrame()

    # Minimum dwell by DP.
    dwell_source = tn[
        tn["dwell_minutes"].notna()
        & (tn["dwell_minutes"] >= 0)
    ].copy()

    min_dwell_by_dp = (
        dwell_source.groupby("dp_id", observed=True)["dwell_minutes"]
        .min()
        .to_dict()
    )

    # Segment/link time from all repeated runs of this train_name.
    tn = tn.sort_values(["train_label", "arrival_seconds"]).copy()

    tn["next_dp_id"] = tn.groupby("train_label", observed=True)["dp_id"].shift(-1)
    tn["next_dp_name"] = tn.groupby("train_label", observed=True)["dp_name"].shift(-1)
    tn["next_mileage"] = tn.groupby("train_label", observed=True)["mileage"].shift(-1)
    tn["next_arrival_seconds"] = tn.groupby("train_label", observed=True)["arrival_seconds"].shift(-1)

    tn["link_seconds"] = tn["next_arrival_seconds"] - tn["departure_seconds"]

    seg = tn[
        tn["next_dp_id"].notna()
        & tn["link_seconds"].notna()
        & (tn["link_seconds"] >= 0)
    ].copy()

    seg["next_dp_id"] = seg["next_dp_id"].astype(int)

    min_link_by_pair = (
        seg.groupby(["dp_id", "next_dp_id"], observed=True)["link_seconds"]
        .min()
        .to_dict()
    )

    segment_table = (
        seg.groupby(["dp_id", "dp_name", "next_dp_id", "next_dp_name"], observed=True)
        .agg(
            min_link_seconds=("link_seconds", "min"),
            p50_link_seconds=("link_seconds", "median"),
            max_link_seconds=("link_seconds", "max"),
            observed_runs=("train_label", "nunique"),
        )
        .reset_index()
    )

    segment_table["min_link_minutes"] = segment_table["min_link_seconds"] / 60

    # Rebuild TPC profile.
    tpc_rows = []
    current_time = 0.0

    for i in range(len(route)):
        r = route.iloc[i]
        dp_id = int(r["dp_id"])

        dwell_min = float(min_dwell_by_dp.get(dp_id, 0.0))
        dwell_sec = dwell_min * 60.0

        arrival_t = current_time
        departure_t = arrival_t + dwell_sec

        tpc_rows.append(
            {
                "plot_time_seconds": arrival_t,
                "plot_time_hours": arrival_t / 3600,
                "mileage": float(r["mileage"]),
                "dp_id": dp_id,
                "dp_name": str(r["dp_name"]),
                "event": "TPC Arrival",
                "train_name": selected_train_name,
                "tpc_route_template": str(route_label),
                "tpc_dwell_minutes": dwell_min,
            }
        )

        tpc_rows.append(
            {
                "plot_time_seconds": departure_t,
                "plot_time_hours": departure_t / 3600,
                "mileage": float(r["mileage"]),
                "dp_id": dp_id,
                "dp_name": str(r["dp_name"]),
                "event": "TPC Departure",
                "train_name": selected_train_name,
                "tpc_route_template": str(route_label),
                "tpc_dwell_minutes": dwell_min,
            }
        )

        if i < len(route) - 1:
            next_r = route.iloc[i + 1]
            next_dp_id = int(next_r["dp_id"])

            link_sec = min_link_by_pair.get((dp_id, next_dp_id), None)

            if link_sec is None:
                # Fallback to template train's actual link time.
                link_sec = float(next_r["arrival_seconds"] - r["departure_seconds"])

            if pd.isna(link_sec) or link_sec < 0:
                link_sec = 0.0

            current_time = departure_t + link_sec

    tpc = pd.DataFrame(tpc_rows)

    return tpc, route, route_label, segment_table


def plot_tpc_only(tpc_df, selected_train_name):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=tpc_df["plot_time_hours"],
            y=tpc_df["mileage"],
            mode="lines+markers",
            name="TPC/Base",
            line=dict(width=3),
            marker=dict(size=6),
            customdata=tpc_df[["dp_name", "event", "tpc_dwell_minutes"]],
            hovertemplate=(
                "TPC/Base<br>"
                "Time: %{x:.2f} hr<br>"
                "Mileage: %{y:.2f}<br>"
                "DP: %{customdata[0]}<br>"
                "Event: %{customdata[1]}<br>"
                "Min dwell: %{customdata[2]:.2f} min"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=f"TPC/Base Stringline — {selected_train_name}",
        xaxis_title="TPC Time from First Event (hours)",
        yaxis_title="Mileage",
        height=720,
        hovermode="closest",
    )

    return fig


def plot_actual_vs_tpc(actual_plot_df, tpc_df, selected_train_label):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=actual_plot_df["plot_time_hours"],
            y=actual_plot_df["mileage"],
            mode="lines+markers",
            name="Actual Train Run",
            line=dict(width=3),
            marker=dict(size=6),
            customdata=actual_plot_df[
                ["dp_name", "event", "arrival_seconds", "departure_seconds", "dwell_minutes"]
            ],
            hovertemplate=(
                "Actual<br>"
                "Time: %{x:.2f} hr<br>"
                "Mileage: %{y:.2f}<br>"
                "DP: %{customdata[0]}<br>"
                "Event: %{customdata[1]}<br>"
                "Arrival sec: %{customdata[2]}<br>"
                "Departure sec: %{customdata[3]}<br>"
                "Dwell: %{customdata[4]:.2f} min"
                "<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=tpc_df["plot_time_hours"],
            y=tpc_df["mileage"],
            mode="lines",
            name="TPC/Base Overlay",
            line=dict(width=3, dash="dash"),
            customdata=tpc_df[["dp_name", "event", "tpc_dwell_minutes"]],
            hovertemplate=(
                "TPC/Base<br>"
                "Time: %{x:.2f} hr<br>"
                "Mileage: %{y:.2f}<br>"
                "DP: %{customdata[0]}<br>"
                "Event: %{customdata[1]}<br>"
                "Min dwell: %{customdata[2]:.2f} min"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=f"Actual vs TPC/Base — {selected_train_label}",
        xaxis_title="Time from First Event (hours)",
        yaxis_title="Mileage",
        height=720,
        hovermode="closest",
    )

    return fig


# =====================================================
# MAIN MENU
# =====================================================
if st.session_state.active_view == "Main Menu":
    st.subheader("Main Menu")
    st.write("Choose an analysis view. Opening a view clears cached data first to reduce memory issues.")

    c1, c2 = st.columns(2)

    with c1:
        if st.button("1. Stringline", width="stretch"):
            clear_memory()
            st.session_state.active_view = "Stringline"
            st.rerun()

        if st.button("2. Train Performance Table", width="stretch"):
            clear_memory()
            st.session_state.active_view = "Train Performance Table"
            st.rerun()

        if st.button("5. Never Dispatched Trains", width="stretch"):
            clear_memory()
            st.session_state.active_view = "Never Dispatched Trains"
            st.rerun()

    with c2:
        if st.button("3. Speed Distribution by Train Name", width="stretch"):
            clear_memory()
            st.session_state.active_view = "Speed Distribution by Train Name"
            st.rerun()

        if st.button("4. Average Delay by DP and Train Group", width="stretch"):
            clear_memory()
            st.session_state.active_view = "Average Delay by DP and Train Group"
            st.rerun()

        if st.button("6. TPC/Base Run Comparison", width="stretch"):
            clear_memory()
            st.session_state.active_view = "TPC/Base Run Comparison"
            st.rerun()

    st.stop()


# =====================================================
# VIEW HEADER
# =====================================================
if st.button("Back to Main Menu / Clear Memory", width="stretch"):
    clear_memory()
    st.session_state.active_view = "Main Menu"
    st.rerun()

st.divider()

df, delay_min_cols = load_and_prepare()

# Keep a separate bad-record dataframe for the new view.
never_dispatched_df = df[df["never_dispatched_record"]].copy()

# Exclude bad records from all normal analyses.
analysis_df = df[~df["never_dispatched_record"]].copy()

st.success(f"Loaded: {selected_file.name}")

c1, c2, c3, c4, c5, c6 = st.columns(6)

c1.metric("Rows", f"{len(df):,}")
c2.metric("Analysis Rows", f"{len(analysis_df):,}")
c3.metric("Excluded Records", f"{len(never_dispatched_df):,}")
c4.metric("Train Runs", f"{analysis_df['train_label'].nunique():,}")
c5.metric("Train Names", f"{analysis_df['train_name'].nunique():,}")
c6.metric("Decision Points", f"{analysis_df['dp_id'].nunique():,}")

with st.expander("Debug / file info"):
    st.write("Data folder:", DATA_DIR.resolve())
    st.write("Loaded file:", selected_file.name)
    st.write("Columns loaded:", list(df.columns))
    st.write("Memory MB:", round(df.memory_usage(deep=True).sum() / 1024 / 1024, 1))
    st.write("Mileage min:", float(analysis_df["mileage"].min()) if not analysis_df.empty else None)
    st.write("Mileage max:", float(analysis_df["mileage"].max()) if not analysis_df.empty else None)
    st.write("Arrival hour min:", float(analysis_df["arrival_hour"].min()) if not analysis_df.empty else None)
    st.write("Arrival hour max:", float(analysis_df["arrival_hour"].max()) if not analysis_df.empty else None)
    st.write("Excluded condition: departure_seconds == 0 and arrival_seconds > 0")

# TPC page uses the full valid analysis_df.
# Other pages can use common sidebar filters.
if st.session_state.active_view == "TPC/Base Run Comparison":
    base_df = analysis_df
else:
    base_df = common_filters(analysis_df)


# =====================================================
# 1. STRINGLINE
# =====================================================
if st.session_state.active_view == "Stringline":
    st.header("Stringline")

    st.caption(
        "Passenger and freight are differentiated by color. Each train run is drawn as its own line. "
        "Records with departure_seconds = 0 and arrival_seconds > 0 are excluded."
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

    global_min_hour = float(analysis_df["arrival_hour"].min())
    global_max_hour = float(analysis_df["arrival_hour"].max())

    WINDOW_HOURS = 24.0

    max_start_hour = max(global_min_hour, global_max_hour - WINDOW_HOURS)

    slider_min = int(math.floor(global_min_hour))
    slider_max = int(math.ceil(max_start_hour))

    if slider_max <= slider_min:
        slider_max = slider_min + 1

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
            "Use train group, train name, or train run filters if needed."
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

    full_min_mile = float(analysis_df["mileage"].min())
    full_max_mile = float(analysis_df["mileage"].max())

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
        safe_dataframe(
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
            ],
            max_rows=3000
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

    perf_df = base_df[
        [
            "train_label",
            "generated_train_id",
            "train_name",
            "train_type",
            "arrival_seconds",
            "departure_seconds",
            "mileage",
            "datapoint_id",
            "dwell_minutes",
            delay_col,
        ]
    ].copy()

    run_summary = (
        perf_df.groupby(
            ["train_label", "generated_train_id", "train_name", "train_type"],
            dropna=False,
            observed=True
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

    del perf_df
    gc.collect()

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

    safe_dataframe(run_summary, max_rows=10000)


# =====================================================
# 3. SPEED DISTRIBUTION BY TRAIN NAME
# =====================================================
elif st.session_state.active_view == "Speed Distribution by Train Name":
    st.header("Speed Distribution by Train Name")

    speed_df = base_df[
        [
            "train_label",
            "generated_train_id",
            "train_name",
            "train_type",
            "arrival_seconds",
            "departure_seconds",
            "mileage",
        ]
    ].copy()

    run_summary = (
        speed_df.groupby(
            ["train_label", "generated_train_id", "train_name", "train_type"],
            dropna=False,
            observed=True
        )
        .agg(
            first_arrival_seconds=("arrival_seconds", "min"),
            last_departure_seconds=("departure_seconds", "max"),
            min_mileage=("mileage", "min"),
            max_mileage=("mileage", "max"),
        )
        .reset_index()
    )

    del speed_df
    gc.collect()

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
        run_summary.groupby(["train_name", "train_type"], dropna=False, observed=True)["avg_speed_mph"]
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
        run_summary.groupby(["train_name", "train_type"], dropna=False, observed=True)
        .size()
        .reset_index(name="train_runs")
    )

    speed_stats = speed_stats.merge(
        run_counts,
        on=["train_name", "train_type"],
        how="left"
    )

    st.subheader("Speed Percentile Table")
    safe_dataframe(speed_stats.sort_values("p50_mph"), max_rows=10000)

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

    MAX_BOXPLOT_ROWS = 30000

    if len(box_df) > MAX_BOXPLOT_ROWS:
        st.warning(f"Boxplot has too many records. Showing first {MAX_BOXPLOT_ROWS:,}.")
        box_df = box_df.head(MAX_BOXPLOT_ROWS)

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
# 4. AVERAGE DELAY BY DP AND TRAIN GROUP
# =====================================================
elif st.session_state.active_view == "Average Delay by DP and Train Group":
    st.header("Average Delay by DP Location and Train Group")

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
            "train_type",
            "dp_id",
            "dp_name",
            "mileage",
            delay_col,
        ]
    ].copy()

    train_dp_delay = (
        delay_df.groupby(
            ["train_label", "train_type", "dp_id", "dp_name", "mileage"],
            dropna=False,
            observed=True
        )
        .agg(
            train_delay_min=(delay_col, "sum")
        )
        .reset_index()
    )

    dp_group_delay = (
        train_dp_delay.groupby(
            ["train_type", "dp_id", "dp_name", "mileage"],
            dropna=False,
            observed=True
        )
        .agg(
            avg_train_delay_min=("train_delay_min", "mean"),
            p50_train_delay_min=("train_delay_min", "median"),
            max_train_delay_min=("train_delay_min", "max"),
            train_runs=("train_label", "nunique"),
        )
        .reset_index()
        .sort_values(["mileage", "train_type"])
    )

    del delay_df
    del train_dp_delay
    gc.collect()

    st.subheader("Average Train Delay by DP")

    if dp_group_delay.empty:
        st.info("No delay data available.")
        st.stop()

    top_n = st.slider(
        "Show top N DP locations by total average delay",
        min_value=10,
        max_value=min(200, dp_group_delay["dp_id"].nunique()),
        value=min(50, dp_group_delay["dp_id"].nunique()),
        step=10
    )

    top_dp_ids = (
        dp_group_delay.groupby(["dp_id", "dp_name", "mileage"], dropna=False, observed=True)
        .agg(total_avg_delay_min=("avg_train_delay_min", "sum"))
        .reset_index()
        .sort_values("total_avg_delay_min", ascending=False)
        .head(top_n)["dp_id"]
        .tolist()
    )

    plot_df = dp_group_delay[dp_group_delay["dp_id"].isin(top_dp_ids)].copy()

    plot_df["dp_label"] = (
        plot_df["mileage"].round(1).astype(str)
        + " | "
        + plot_df["dp_name"].astype(str)
    )

    plot_df = plot_df.sort_values("mileage")

    fig = px.bar(
        plot_df,
        x="dp_label",
        y="avg_train_delay_min",
        color="train_type",
        barmode="stack",
        hover_data=[
            "train_type",
            "dp_id",
            "dp_name",
            "mileage",
            "avg_train_delay_min",
            "p50_train_delay_min",
            "max_train_delay_min",
            "train_runs",
        ],
        title="Stacked Average Train Delay by DP Location",
        labels={
            "dp_label": "DP Location",
            "avg_train_delay_min": "Average Train Delay (min)",
            "train_type": "Train Group",
        },
    )

    fig.update_layout(
        height=720,
        xaxis_tickangle=-60,
        hovermode="closest",
    )

    st.plotly_chart(fig, width="stretch")

    st.subheader("Average Delay Table")
    safe_dataframe(dp_group_delay, max_rows=10000)


# =====================================================
# 5. NEVER DISPATCHED TRAINS
# =====================================================
elif st.session_state.active_view == "Never Dispatched Trains":
    st.header("Never Dispatched Trains")

    st.caption(
        "Records shown here have departure_seconds = 0 while arrival_seconds > 0. "
        "They are excluded from all other analysis views. "
        "For display, departure and dwell are shown as NA."
    )

    if never_dispatched_df.empty:
        st.success("No never-dispatched records found.")
        st.stop()

    display_bad = never_dispatched_df.copy()

    display_bad["departure_seconds"] = pd.NA
    display_bad["departure_ddhhmmss"] = pd.NA
    display_bad["dwell_seconds"] = pd.NA
    display_bad["dwell_minutes"] = pd.NA

    bad_summary = (
        display_bad.groupby(
            ["train_label", "generated_train_id", "train_name", "train_type"],
            dropna=False,
            observed=True
        )
        .agg(
            bad_records=("datapoint_id", "count"),
            first_bad_arrival_seconds=("arrival_seconds", "min"),
            last_bad_arrival_seconds=("arrival_seconds", "max"),
            first_bad_dp=("dp_name", "first"),
            last_bad_dp=("dp_name", "last"),
            min_mileage=("mileage", "min"),
            max_mileage=("mileage", "max"),
        )
        .reset_index()
        .sort_values("bad_records", ascending=False)
    )

    bad_summary["first_bad_arrival"] = bad_summary["first_bad_arrival_seconds"].apply(seconds_to_ddhhmmss)
    bad_summary["last_bad_arrival"] = bad_summary["last_bad_arrival_seconds"].apply(seconds_to_ddhhmmss)

    bad_summary = bad_summary[
        [
            "train_label",
            "generated_train_id",
            "train_name",
            "train_type",
            "bad_records",
            "first_bad_arrival",
            "last_bad_arrival",
            "first_bad_dp",
            "last_bad_dp",
            "min_mileage",
            "max_mileage",
        ]
    ]

    st.subheader("Summary by Train")
    safe_dataframe(bad_summary, max_rows=10000)

    st.subheader("Raw Never-Dispatched Records")

    raw_cols = [
        "train_label",
        "generated_train_id",
        "train_name",
        "train_type",
        "datapoint_id",
        "dp_id",
        "dp_name",
        "mileage",
        "arrival_ddhhmmss",
        "departure_ddhhmmss",
        "arrival_seconds",
        "departure_seconds",
        "dwell_minutes",
        "total_delay_min_all_codes",
        "total_delay_min_cn_filtered",
    ]

    safe_dataframe(
        display_bad[[c for c in raw_cols if c in display_bad.columns]],
        max_rows=20000
    )


# =====================================================
# 6. TPC / BASE RUN COMPARISON
# =====================================================
elif st.session_state.active_view == "TPC/Base Run Comparison":
    st.header("TPC/Base Run Comparison")

    st.caption(
        "This page builds a TPC/base stringline for each train_name using minimum observed dwell time "
        "at each DP and minimum observed link time between consecutive DPs. The actual train run and "
        "TPC/base profile are both normalized to start at 0 seconds."
    )

    tpc_source = analysis_df[
        analysis_df["mileage"].notna()
        & analysis_df["arrival_seconds"].notna()
        & analysis_df["departure_seconds"].notna()
        & (analysis_df["departure_seconds"] >= analysis_df["arrival_seconds"])
        & (analysis_df["departure_seconds"] > 0)
    ].copy()

    if tpc_source.empty:
        st.info("No valid records available for TPC calculation.")
        st.stop()

    train_name_options = sorted(tpc_source["train_name"].astype(str).dropna().unique())

    selected_train_name = st.selectbox(
        "Select train_name",
        train_name_options,
        index=0
    )

    same_name_df = tpc_source[
        tpc_source["train_name"].astype(str) == selected_train_name
    ].copy()

    train_labels_for_name = sorted(
        same_name_df["train_label"].astype(str).dropna().unique()
    )

    if not train_labels_for_name:
        st.info("No train runs found for this train_name.")
        st.stop()

    state_key = f"tpc_label_index_{selected_train_name}"

    if state_key not in st.session_state:
        st.session_state[state_key] = 0

    st.session_state[state_key] = max(
        0,
        min(st.session_state[state_key], len(train_labels_for_name) - 1)
    )

    cprev, csel, cnext = st.columns([1, 4, 1])

    with cprev:
        if st.button("Previous train run", width="stretch"):
            st.session_state[state_key] = max(0, st.session_state[state_key] - 1)
            st.rerun()

    with cnext:
        if st.button("Next train run", width="stretch"):
            st.session_state[state_key] = min(
                len(train_labels_for_name) - 1,
                st.session_state[state_key] + 1
            )
            st.rerun()

    with csel:
        selected_train_label = st.selectbox(
            "Selected train_label for comparison",
            train_labels_for_name,
            index=st.session_state[state_key]
        )

        st.session_state[state_key] = train_labels_for_name.index(selected_train_label)

    tpc_df, route_template_df, route_template_label, segment_table = build_tpc_profile_for_train_name(
        tpc_source,
        selected_train_name
    )

    if tpc_df.empty:
        st.info("Could not build TPC profile for this train_name.")
        st.stop()

    actual_run = same_name_df[
        same_name_df["train_label"].astype(str) == selected_train_label
    ].copy()

    actual_run = actual_run.sort_values("arrival_seconds")

    if actual_run.empty:
        st.info("No actual records found for selected train_label.")
        st.stop()

    actual_t0 = actual_run["arrival_seconds"].min()

    actual_plot = make_two_point_stringline_rows(
        actual_run,
        time_prefix=actual_t0
    )

    actual_duration_hr = (
        actual_run["departure_seconds"].max()
        - actual_run["arrival_seconds"].min()
    ) / 3600

    tpc_duration_hr = tpc_df["plot_time_hours"].max()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Train name", selected_train_name)
    c2.metric("Runs found", f"{len(train_labels_for_name):,}")
    c3.metric("TPC duration hr", f"{tpc_duration_hr:.2f}")
    c4.metric("Actual duration hr", f"{actual_duration_hr:.2f}")

    st.write(f"TPC route template used: `{route_template_label}`")

    left, right = st.columns(2)

    full_min_mile = float(tpc_source["mileage"].min())
    full_max_mile = float(tpc_source["mileage"].max())

    with left:
        st.subheader("TPC/Base Stringline")
        fig_tpc = plot_tpc_only(tpc_df, selected_train_name)

        fig_tpc.update_yaxes(
            range=[full_min_mile, full_max_mile],
            title="Mileage"
        )

        st.plotly_chart(fig_tpc, width="stretch")

    with right:
        st.subheader("Actual Train Run vs TPC/Base")
        fig_compare = plot_actual_vs_tpc(
            actual_plot,
            tpc_df,
            selected_train_label
        )

        fig_compare.update_yaxes(
            range=[full_min_mile, full_max_mile],
            title="Mileage"
        )

        st.plotly_chart(fig_compare, width="stretch")

    with st.expander("TPC profile table"):
        safe_dataframe(
            tpc_df[
                [
                    "plot_time_hours",
                    "mileage",
                    "dp_id",
                    "dp_name",
                    "event",
                    "tpc_dwell_minutes",
                    "tpc_route_template",
                ]
            ],
            max_rows=5000
        )

    with st.expander("Minimum link-time table"):
        if segment_table.empty:
            st.info("No segment table available.")
        else:
            safe_dataframe(
                segment_table[
                    [
                        "dp_id",
                        "dp_name",
                        "next_dp_id",
                        "next_dp_name",
                        "min_link_minutes",
                        "p50_link_seconds",
                        "max_link_seconds",
                        "observed_runs",
                    ]
                ],
                max_rows=5000
            )

    with st.expander("Actual selected train run table"):
        safe_dataframe(
            actual_run[
                [
                    "train_label",
                    "train_name",
                    "train_type",
                    "dp_id",
                    "dp_name",
                    "mileage",
                    "arrival_ddhhmmss",
                    "departure_ddhhmmss",
                    "dwell_minutes",
                    "total_delay_min_all_codes",
                    "total_delay_min_cn_filtered",
                ]
            ],
            max_rows=5000
        )