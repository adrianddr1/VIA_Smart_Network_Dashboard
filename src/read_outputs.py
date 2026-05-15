# -*- coding: utf-8 -*-
"""
Created on Thu May 14 23:31:22 2026

@author: ZhaoJ
"""

import streamlit as st
import json
import re
import pandas as pd
import plotly.express as px
from st_aggrid import AgGrid, GridOptionsBuilder

st.set_page_config(layout="wide")
st.title("Smart Network Output Viewer — Datapoint / Delay Join")

uploaded_file = st.file_uploader("Upload Output JSON", type="json")


def duration_to_seconds(value):
    if not isinstance(value, str):
        return None

    m = re.match(r"P(\d+)DT(\d+)H(\d+)M(\d+)S", value)
    if not m:
        return None

    d, h, mnt, s = map(int, m.groups())
    return d * 86400 + h * 3600 + mnt * 60 + s


def ddhhmmss_to_seconds(value):
    if not isinstance(value, str):
        return None

    m = re.match(r"^(\d+):(\d{1,2}):(\d{1,2}):(\d{1,2})$", value.strip())
    if not m:
        return None

    d, h, mnt, s = map(int, m.groups())
    return d * 86400 + h * 3600 + mnt * 60 + s


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


def train_type_from_name(name):
    if isinstance(name, str) and name.startswith("P"):
        return "Passenger"
    return "Freight / Other"


def show_aggrid(df, key, height=600, page_size=100):
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_default_column(
        sortable=True,
        filter=True,
        resizable=True,
        editable=False,
    )
    gb.configure_grid_options(
        pagination=True,
        paginationPageSize=page_size,
        enableRangeSelection=True,
    )

    AgGrid(
        df,
        gridOptions=gb.build(),
        height=height,
        fit_columns_on_grid_load=False,
        key=key,
    )


def find_delay_table(sim_results):
    """
    Find the delay table in sim_results.
    Assumes the delay table has schema/data and path or field names contain 'delay'.
    """
    candidates = []

    def walk(obj, path="sim_results"):
        if isinstance(obj, dict):
            if "schema" in obj and "data" in obj:
                fields = obj.get("schema", {}).get("fields", [])
                field_names = [f.get("name", "") for f in fields]

                if "delay" in path.lower() or any("delay" in x.lower() for x in field_names):
                    candidates.append((path, obj))

            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    walk(v, f"{path}.{k}")

        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                if isinstance(item, (dict, list)):
                    walk(item, f"{path}[{i}]")

    walk(sim_results)

    if not candidates:
        return None, None

    return candidates[0]


if uploaded_file:
    raw = json.load(uploaded_file)
    sim_results = raw.get("sim_results", {})

    # -----------------------------
    # 1. First chunk: datapoints
    # -----------------------------
    datapoint_rows = sim_results.get("datapoints", {}).get("data", [])

    if not datapoint_rows:
        st.error("No sim_results.datapoints.data found.")
        st.stop()

    datapoints = pd.DataFrame(
        datapoint_rows,
        columns=[
            "datapoint_id",
            "generated_train_id",
            "dp_id",
            "link_id",
            "arrival_raw",
            "departure_raw",
        ],
    )

    datapoints["datapoint_id"] = datapoints["datapoint_id"].astype("int64")
    datapoints["generated_train_id"] = datapoints["generated_train_id"].astype("int64")
    datapoints["dp_id"] = datapoints["dp_id"].astype("int64")
    datapoints["link_id"] = datapoints["link_id"].astype("int64")

    datapoints["arrival_seconds"] = datapoints["arrival_raw"].apply(duration_to_seconds)
    datapoints["departure_seconds"] = datapoints["departure_raw"].apply(duration_to_seconds)

    datapoints["arrival_hour"] = datapoints["arrival_seconds"] / 3600
    datapoints["departure_hour"] = datapoints["departure_seconds"] / 3600

    datapoints["dwell_minutes"] = (
        datapoints["departure_seconds"] - datapoints["arrival_seconds"]
    ) / 60

    datapoints["arrival_time"] = datapoints["arrival_seconds"].apply(seconds_to_ddhhmmss)
    datapoints["departure_time"] = datapoints["departure_seconds"].apply(seconds_to_ddhhmmss)

    # -----------------------------
    # Generated train names
    # -----------------------------
    generated_rows = sim_results.get("generated_trains", {}).get("data", [])

    if generated_rows:
        generated_trains = pd.DataFrame(
            generated_rows,
            columns=["generated_train_id", "parent_train_id", "train_name"],
        )

        generated_trains["generated_train_id"] = generated_trains["generated_train_id"].astype("int64")
        generated_trains["parent_train_id"] = generated_trains["parent_train_id"].astype("int64")

        datapoints = datapoints.merge(
            generated_trains,
            on="generated_train_id",
            how="left",
        )
    else:
        generated_trains = pd.DataFrame()
        datapoints["parent_train_id"] = pd.NA
        datapoints["train_name"] = ""

    datapoints["train_name"] = datapoints["train_name"].fillna("")
    datapoints["train_type"] = datapoints["train_name"].apply(train_type_from_name)
    datapoints["train_label"] = (
        datapoints["generated_train_id"].astype(str)
        + " | "
        + datapoints["train_name"]
    )

    # -----------------------------
    # 2. Third chunk: delays
    # -----------------------------
    delay_path, delay_obj = find_delay_table(sim_results)

    if delay_obj is None:
        st.error("No delay table found.")
        st.stop()

    delay_rows = delay_obj.get("data", [])

    delays = pd.DataFrame(
        delay_rows,
        columns=[
            "delay_id",
            "datapoint_id",
            "delay_code",
            "delay_raw",
        ],
    )

    delays["delay_id"] = pd.to_numeric(delays["delay_id"], errors="coerce")
    delays["datapoint_id"] = pd.to_numeric(delays["datapoint_id"], errors="coerce")
    delays["delay_code"] = pd.to_numeric(delays["delay_code"], errors="coerce")

    # IMPORTANT:
    # Do NOT assume unit. Keep raw, and provide two interpretations.
    delays["delay_numeric"] = pd.to_numeric(delays["delay_raw"], errors="coerce")
    delays["delay_minutes_if_hours"] = delays["delay_numeric"] * 60
    delays["delay_minutes_if_minutes"] = delays["delay_numeric"]

    delays["delay_code_group"] = delays["delay_code"].fillna(0).astype(int) & 0xFF00
    delays["delay_code_hex"] = delays["delay_code"].apply(
        lambda x: hex(int(x)) if pd.notna(x) else ""
    )
    delays["delay_code_group_hex"] = delays["delay_code_group"].apply(
        lambda x: hex(int(x)) if pd.notna(x) else ""
    )
    delays["included_by_cn_filter"] = ~delays["delay_code_group"].isin(
        [0x300, 0x500, 0x600]
    )

    # -----------------------------
    # 3. Join: datapoints.datapoint_id = delays.datapoint_id
    # -----------------------------
    joined = delays.merge(
        datapoints,
        on="datapoint_id",
        how="left",
    )

    # -----------------------------
    # Basic app summary
    # -----------------------------
    st.success("Output file loaded and datapoints/delays joined")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Datapoints", f"{len(datapoints):,}")
    c2.metric("Delay Records", f"{len(delays):,}")
    c3.metric("Joined Delay Records", f"{len(joined):,}")
    c4.metric("Delay Table", delay_path)

    view = st.sidebar.radio(
        "Select View",
        [
            "1. Train Stringlines",
            "2. Datapoint + Delay Join",
            "3. DP Summary",
            "4. Delay Code Distribution",
            "5. Raw Review",
        ],
    )

    # =========================================================
    # 1. Train Stringlines
    # =========================================================
    if view == "1. Train Stringlines":
        st.header("Train Stringlines")

        train_type_filter = st.selectbox(
            "Train type",
            ["All", "Passenger", "Freight / Other"],
        )

        plot_df = datapoints.copy()

        if train_type_filter != "All":
            plot_df = plot_df[plot_df["train_type"] == train_type_filter]

        train_labels = sorted(plot_df["train_label"].dropna().unique())

        selected_trains = st.multiselect(
            "Select trains",
            train_labels,
            default=train_labels[:10],
        )

        default_start = seconds_to_ddhhmmss(plot_df["arrival_seconds"].min())
        default_end = seconds_to_ddhhmmss(
            min(plot_df["arrival_seconds"].min() + 24 * 3600, plot_df["arrival_seconds"].max())
        )

        c1, c2 = st.columns(2)
        start_text = c1.text_input("Start time DD:HH:MM:SS", value=default_start)
        end_text = c2.text_input("End time DD:HH:MM:SS", value=default_end)

        start_seconds = ddhhmmss_to_seconds(start_text)
        end_seconds = ddhhmmss_to_seconds(end_text)

        if start_seconds is None or end_seconds is None:
            st.error("Invalid time format. Use DD:HH:MM:SS.")
            st.stop()

        filtered = plot_df[
            (plot_df["train_label"].isin(selected_trains))
            & (plot_df["arrival_seconds"] >= start_seconds)
            & (plot_df["arrival_seconds"] <= end_seconds)
        ].copy()

        chart_df = filtered.head(50000)

        fig = px.line(
            chart_df.sort_values(["train_label", "arrival_seconds"]),
            x="arrival_hour",
            y="dp_id",
            color="train_label",
            markers=True,
            hover_data=[
                "datapoint_id",
                "generated_train_id",
                "train_name",
                "train_type",
                "dp_id",
                "link_id",
                "arrival_time",
                "departure_time",
                "dwell_minutes",
            ],
            title="Train Movement Stringline",
            labels={
                "arrival_hour": "Arrival Time from Simulation Start (hours)",
                "dp_id": "Decision Point ID",
            },
        )

        st.plotly_chart(fig, use_container_width=True)

    # =========================================================
    # 2. Datapoint + Delay Join
    # =========================================================
    elif view == "2. Datapoint + Delay Join":
        st.header("Datapoint + Delay Join")

        st.write(
            "This directly joins third chunk delay records to first chunk datapoints using "
            "`delays.datapoint_id = datapoints.datapoint_id`."
        )

        cols = [
            "delay_id",
            "datapoint_id",
            "generated_train_id",
            "train_name",
            "train_type",
            "dp_id",
            "link_id",
            "arrival_time",
            "departure_time",
            "dwell_minutes",
            "delay_code",
            "delay_code_hex",
            "delay_code_group_hex",
            "delay_raw",
            "delay_minutes_if_hours",
            "delay_minutes_if_minutes",
            "included_by_cn_filter",
        ]

        show_aggrid(
            joined[cols].head(2000),
            key="joined_grid",
            height=650,
            page_size=100,
        )

    # =========================================================
    # 3. DP Summary
    # =========================================================
    elif view == "3. DP Summary":
        st.header("Decision Point Summary")

        unit_choice = st.radio(
            "Delay duration unit assumption",
            ["Use raw delay as hours", "Use raw delay as minutes"],
            horizontal=True,
        )

        delay_col = (
            "delay_minutes_if_hours"
            if unit_choice == "Use raw delay as hours"
            else "delay_minutes_if_minutes"
        )

        dp_dwell = (
            datapoints.groupby("dp_id")
            .agg(
                datapoints=("datapoint_id", "count"),
                trains=("generated_train_id", "nunique"),
                dwell_events=("dwell_minutes", lambda x: (x > 0).sum()),
                total_dwell_min=("dwell_minutes", "sum"),
                max_dwell_min=("dwell_minutes", "max"),
            )
            .reset_index()
        )

        dp_delay_all = (
            joined.groupby("dp_id")
            .agg(
                delay_records=("delay_id", "count"),
                total_delay_min_all_codes=(delay_col, "sum"),
                max_delay_min=(delay_col, "max"),
            )
            .reset_index()
        )

        dp_delay_cn = (
            joined[joined["included_by_cn_filter"]]
            .groupby("dp_id")
            .agg(
                delay_records_cn_filtered=("delay_id", "count"),
                total_delay_min_cn_filtered=(delay_col, "sum"),
            )
            .reset_index()
        )

        dp_summary = (
            dp_dwell
            .merge(dp_delay_all, on="dp_id", how="left")
            .merge(dp_delay_cn, on="dp_id", how="left")
        )

        fill_cols = [
            "delay_records",
            "total_delay_min_all_codes",
            "max_delay_min",
            "delay_records_cn_filtered",
            "total_delay_min_cn_filtered",
        ]

        for c in fill_cols:
            dp_summary[c] = dp_summary[c].fillna(0)

        dp_summary["delay_minus_dwell_all_codes"] = (
            dp_summary["total_delay_min_all_codes"] - dp_summary["total_dwell_min"]
        )

        dp_summary = dp_summary.sort_values("total_delay_min_all_codes", ascending=False)

        show_aggrid(
            dp_summary,
            key="dp_summary_grid",
            height=650,
            page_size=100,
        )

    # =========================================================
    # 4. Delay Code Distribution
    # =========================================================
    elif view == "4. Delay Code Distribution":
        st.header("Delay Code Distribution")

        unit_choice = st.radio(
            "Delay duration unit assumption",
            ["Use raw delay as hours", "Use raw delay as minutes"],
            horizontal=True,
        )

        delay_col = (
            "delay_minutes_if_hours"
            if unit_choice == "Use raw delay as hours"
            else "delay_minutes_if_minutes"
        )

        delay_code_summary = (
            joined.groupby(
                [
                    "delay_code",
                    "delay_code_hex",
                    "delay_code_group",
                    "delay_code_group_hex",
                    "included_by_cn_filter",
                ]
            )
            .agg(
                delay_records=("delay_id", "count"),
                total_delay_min=(delay_col, "sum"),
                max_delay_min=(delay_col, "max"),
                affected_datapoints=("datapoint_id", "nunique"),
                affected_trains=("generated_train_id", "nunique"),
                affected_dps=("dp_id", "nunique"),
            )
            .reset_index()
            .sort_values("total_delay_min", ascending=False)
        )

        show_aggrid(
            delay_code_summary,
            key="delay_code_summary_grid",
            height=500,
            page_size=100,
        )

        fig = px.bar(
            delay_code_summary.head(30),
            x="delay_code_hex",
            y="total_delay_min",
            color="included_by_cn_filter",
            hover_data=[
                "delay_code_group_hex",
                "delay_records",
                "affected_trains",
                "affected_dps",
            ],
            title="Top Delay Codes by Total Delay Minutes",
            labels={
                "delay_code_hex": "Delay Code",
                "total_delay_min": "Total Delay Minutes",
            },
        )

        st.plotly_chart(fig, use_container_width=True)

    # =========================================================
    # 5. Raw Review
    # =========================================================
    elif view == "5. Raw Review":
        st.header("Raw Review")

        st.subheader("First Chunk: Datapoints — first 20 rows")
        st.dataframe(datapoints.head(20), use_container_width=True)

        st.subheader("Generated Trains — first 20 rows")
        st.dataframe(generated_trains.head(20), use_container_width=True)

        st.subheader("Third Chunk: Delays — first 20 rows")
        st.dataframe(delays.head(20), use_container_width=True)

        st.subheader("Joined Datapoint + Delay — first 20 rows")
        st.dataframe(joined.head(20), use_container_width=True)

else:
    st.info("Upload the output JSON file to begin.")