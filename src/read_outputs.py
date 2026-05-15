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


st.set_page_config(layout="wide")
st.title("VIA Smart Network Stringline Dashboard")


# =====================================================
# CONFIG
# =====================================================
DATA_DIR =  Path(__file__).resolve().parents[1] / "inputs"

parquet_files = sorted(DATA_DIR.glob("*.parquet"))

if not parquet_files:
    st.error(f"No parquet files found in: {DATA_DIR.resolve()}")
    st.stop()


# =====================================================
# HELPERS
# =====================================================
@st.cache_data(show_spinner=True)
def load_parquet(path):
    return pd.read_parquet(path)


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

    d, h, m, s = map(int, m.groups())

    return d * 86400 + h * 3600 + m * 60 + s


# =====================================================
# FILE SELECTION
# =====================================================
selected_file = st.sidebar.selectbox(
    "Select parquet file",
    parquet_files,
    format_func=lambda p: p.name
)

df = load_parquet(selected_file)

st.success(f"Loaded: {selected_file.name}")

required_cols = [
    "train_name",
    "dp_name",
    "mileage",
    "arrival_seconds",
    "departure_seconds",
]

missing = [c for c in required_cols if c not in df.columns]

if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()


# =====================================================
# CLEAN DATA
# =====================================================
df = df.copy()

df["train_name"] = df["train_name"].fillna("")
df["dp_name"] = df["dp_name"].fillna("")
df["mileage"] = pd.to_numeric(df["mileage"], errors="coerce")
df["arrival_seconds"] = pd.to_numeric(df["arrival_seconds"], errors="coerce")
df["departure_seconds"] = pd.to_numeric(df["departure_seconds"], errors="coerce")

df["arrival_hour"] = df["arrival_seconds"] / 3600
df["departure_hour"] = df["departure_seconds"] / 3600

df["arrival_time_ddhhmmss"] = df["arrival_seconds"].apply(seconds_to_ddhhmmss)
df["departure_time_ddhhmmss"] = df["departure_seconds"].apply(seconds_to_ddhhmmss)

if "train_type" not in df.columns:
    df["train_type"] = df["train_name"].apply(
        lambda x: "Passenger" if isinstance(x, str) and x.startswith("P") else "Freight / Other"
    )


# =====================================================
# SIDEBAR FILTERS
# =====================================================
st.sidebar.header("Filters")

train_type = st.sidebar.selectbox(
    "Train type",
    ["All", "Passenger", "Freight / Other"]
)

plot_df = df.copy()

if train_type != "All":
    plot_df = plot_df[plot_df["train_type"] == train_type]

train_names = sorted(plot_df["train_name"].dropna().unique())

selected_trains = st.sidebar.multiselect(
    "Select train_name",
    train_names,
    default=train_names[:10]
)

min_time = plot_df["arrival_seconds"].min()
max_time = plot_df["arrival_seconds"].max()

default_start = seconds_to_ddhhmmss(min_time)
default_end = seconds_to_ddhhmmss(min(min_time + 24 * 3600, max_time))

start_text = st.sidebar.text_input(
    "Start time DD:HH:MM:SS",
    value=default_start
)

end_text = st.sidebar.text_input(
    "End time DD:HH:MM:SS",
    value=default_end
)

start_sec = ddhhmmss_to_seconds(start_text)
end_sec = ddhhmmss_to_seconds(end_text)

if start_sec is None or end_sec is None:
    st.error("Invalid time format. Use DD:HH:MM:SS, e.g. 000:00:00:00.")
    st.stop()


# =====================================================
# FILTER DATA
# =====================================================
plot_df = plot_df[
    (plot_df["train_name"].isin(selected_trains))
    & (plot_df["arrival_seconds"] >= start_sec)
    & (plot_df["arrival_seconds"] <= end_sec)
    & (plot_df["mileage"].notna())
].copy()

plot_df = plot_df.sort_values(["train_name", "arrival_seconds"])

st.write(f"Showing {len(plot_df):,} rows")

if len(plot_df) > 80000:
    st.warning("Large chart. Showing first 80,000 rows. Narrow the time/train filter for better speed.")
    plot_df = plot_df.head(80000)


# =====================================================
# STRINGLINE
# =====================================================
st.header("Stringline")

fig = px.line(
    plot_df,
    x="arrival_hour",
    y="mileage",
    color="train_name",
    markers=True,
    hover_data=[
        "train_name",
        "dp_name",
        "mileage",
        "arrival_time_ddhhmmss",
        "departure_time_ddhhmmss",
    ],
    labels={
        "arrival_hour": "Simulation Time (hours)",
        "mileage": "Mileage",
        "train_name": "Train",
    },
    title="Train Stringline by Mileage"
)

# Default visible window = selected input range, normally first 24 hrs
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
    height=750,
    hovermode="closest",
    legend_title_text="Train Name",
)

st.plotly_chart(fig, use_container_width=True)


# =====================================================
# OPTIONAL TABLE
# =====================================================
with st.expander("View filtered data"):
    st.dataframe(
        plot_df[
            [
                "train_name",
                "train_type",
                "dp_name",
                "mileage",
                "arrival_time_ddhhmmss",
                "departure_time_ddhhmmss",
            ]
        ].head(5000),
        use_container_width=True
    )