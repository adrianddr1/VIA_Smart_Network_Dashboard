# -*- coding: utf-8 -*-
"""
Created on Fri May 15 00:34:42 2026

@author: ZhaoJ
"""
import json
import re
from pathlib import Path
import pandas as pd


# =====================================================
# FILE PATHS
# =====================================================
OUTPUT_FILE = Path(
    r"C:\Users\ZhaoJ\OneDrive - DB E.C.O. North America\Desktop\VIA Dashboards\Inputs\TOR_SAR_KMK_WIN_2019_base_traffic_CTA-input-UC-1-output-RT_F0_P0_.json"
)

DP_FILE = Path(
    r"C:\Users\ZhaoJ\OneDrive - DB E.C.O. North America\Desktop\VIA Dashboards\Inputs\TOR_SAR_KMK_WIN_2019_base_traffic_CTA-input-UC-1.json"
)

RESULTS_PATH = Path(
    r"C:\Users\ZhaoJ\OneDrive - DB E.C.O. North America\Desktop\VIA Dashboards\processed"
)

RESULTS_PATH.mkdir(exist_ok=True)


# =====================================================
# HELPER FUNCTIONS
# =====================================================
def duration_to_seconds(value):
    """
    Convert Smart Network duration:
    P000DT05H10M45S
    to seconds.
    """
    if not isinstance(value, str):
        return None

    m = re.match(r"P(\d+)DT(\d+)H(\d+)M(\d+)S", value)

    if not m:
        return None

    days, hours, minutes, seconds = map(int, m.groups())

    return (
        days * 86400
        + hours * 3600
        + minutes * 60
        + seconds
    )


def table_from_schema(block):
    """
    Convert schema + data JSON table into dataframe.
    """
    fields = block.get("schema", {}).get("fields", [])
    data = block.get("data", [])

    columns = [f["name"] for f in fields]

    return pd.DataFrame(data, columns=columns)


def optimize_types(df):
    """
    Reduce parquet size by using smaller numeric types and categories.
    """
    int_cols = [
        "datapoint_id",
        "generated_train_id",
        "parent_train_id",
        "dp_id",
        "link_id",
    ]

    float_cols = [
        "arrival_seconds",
        "departure_seconds",
        "dwell_seconds",
        "dwell_minutes",
        "arrival_hour",
        "mileage",
    ]

    category_cols = [
        "train_name",
        "train_type",
        "train_label",
        "dp_name",
    ]

    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int32")

    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

    for col in category_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

    for col in df.columns:
        if col.startswith("delay_code_") or col.startswith("delay_code_group_"):
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int32")

        if col.startswith("delay_minutes_"):
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

        if col.startswith("included_by_cn_filter_"):
            df[col] = (
                df[col]
                .astype(str)
                .str.upper()
                .map({"TRUE": True, "FALSE": False, "1": True, "0": False})
                .astype("boolean")
            )

    return df


# =====================================================
# MAIN PROCESS
# =====================================================
def process_file(output_json_path, dp_json_path):

    print(f"Reading output JSON: {output_json_path}")

    with open(output_json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    sim = raw["sim_results"]

    # =====================================================
    # CHUNK 1: DATAPOINTS
    # =====================================================
    print("Reading datapoints...")

    datapoints = table_from_schema(sim["datapoints"])

    datapoints["datapoint_id"] = pd.to_numeric(
        datapoints["datapoint_id"],
        errors="coerce"
    )

    datapoints["generated_train_id"] = pd.to_numeric(
        datapoints["generated_train_id"],
        errors="coerce"
    )

    datapoints["dp_id"] = pd.to_numeric(
        datapoints["dp_id"],
        errors="coerce"
    )

    datapoints["link_id"] = pd.to_numeric(
        datapoints["link_id"],
        errors="coerce"
    )

    datapoints["arrival_seconds"] = datapoints["arrival_time"].apply(
        duration_to_seconds
    )

    datapoints["departure_seconds"] = datapoints["departure_time"].apply(
        duration_to_seconds
    )

    datapoints["dwell_seconds"] = (
        datapoints["departure_seconds"]
        - datapoints["arrival_seconds"]
    )

    datapoints["dwell_minutes"] = (
        datapoints["dwell_seconds"] / 60
    )

    datapoints["arrival_hour"] = (
        datapoints["arrival_seconds"] / 3600
    )

    # Remove raw time strings to shrink output
    datapoints = datapoints[
        [
            "datapoint_id",
            "generated_train_id",
            "dp_id",
            "link_id",
            "arrival_seconds",
            "departure_seconds",
            "dwell_seconds",
            "dwell_minutes",
            "arrival_hour",
        ]
    ]

    # =====================================================
    # CHUNK 2: GENERATED TRAINS
    # =====================================================
    print("Reading generated trains...")

    trains = table_from_schema(sim["generated_trains"])

    trains = trains.rename(columns={
        "name": "train_name"
    })

    trains["generated_train_id"] = pd.to_numeric(
        trains["generated_train_id"],
        errors="coerce"
    )

    trains["parent_train_id"] = pd.to_numeric(
        trains["parent_train_id"],
        errors="coerce"
    )

    trains["train_type"] = trains["train_name"].apply(
        lambda x: "Passenger"
        if isinstance(x, str) and x.startswith("P")
        else "Freight / Other"
    )

    trains = trains[
        [
            "generated_train_id",
            "parent_train_id",
            "train_name",
            "train_type",
        ]
    ]

    # =====================================================
    # CHUNK 3: DELAYS
    # =====================================================
    print("Reading delays...")

    delays = table_from_schema(sim["delays"])

    delays["delay_id"] = pd.to_numeric(
        delays["delay_id"],
        errors="coerce"
    )

    delays["datapoint_id"] = pd.to_numeric(
        delays["datapoint_id"],
        errors="coerce"
    )

    delays["delay_code"] = pd.to_numeric(
        delays["delay_code"],
        errors="coerce"
    )

    delays["delay_seconds"] = delays["duration"].apply(
        duration_to_seconds
    )

    delays["delay_minutes"] = (
        delays["delay_seconds"] / 60
    )

    delays["delay_code_group"] = (
        delays["delay_code"].fillna(0).astype(int) & 0xFF00
    )

    delays["included_by_cn_filter"] = ~delays["delay_code_group"].isin(
        [0x300, 0x500, 0x600]
    )

    # Keep only compact useful delay fields
    delays = delays[
        [
            "delay_id",
            "datapoint_id",
            "delay_code",
            "delay_code_group",
            "delay_minutes",
            "included_by_cn_filter",
        ]
    ]

    # =====================================================
    # DP INFO FROM INPUT JSON
    # =====================================================
    print(f"Reading DP file: {dp_json_path}")

    with open(dp_json_path, "r", encoding="utf-8") as f:
        dp_raw = json.load(f)

    dp_block = dp_raw["scenario"]["dps"]["dp"]

    dp_df = table_from_schema(dp_block)

    dp_df = dp_df.rename(columns={
        "name": "dp_name"
    })

    dp_df["dp_id"] = pd.to_numeric(
        dp_df["dp_id"],
        errors="coerce"
    )

    dp_df["mileage"] = pd.to_numeric(
        dp_df["mileage"],
        errors="coerce"
    )

    dp_df = dp_df[
        [
            "dp_id",
            "dp_name",
            "mileage",
        ]
    ]

    # =====================================================
    # JOIN DATAPOINTS + TRAINS + DP INFO
    # =====================================================
    print("Joining datapoints with train names...")

    base = datapoints.merge(
        trains,
        on="generated_train_id",
        how="left"
    )

    print("Joining datapoints with DP names and mileage...")

    base = base.merge(
        dp_df,
        on="dp_id",
        how="left"
    )

    # =====================================================
    # WIDE DELAY FORMAT
    # Only keep:
    # delay_code_1, delay_code_group_1,
    # delay_minutes_1, included_by_cn_filter_1
    # =====================================================
    print("Pivoting delays to compact wide format...")

    delays_sorted = delays.sort_values(
        [
            "datapoint_id",
            "delay_id"
        ]
    ).copy()

    delays_sorted["delay_seq"] = (
        delays_sorted
        .groupby("datapoint_id")
        .cumcount()
        + 1
    )

    # Optional: cap number of delay records per datapoint
    # Keeps files smaller and prevents many rare columns.
    MAX_DELAY_EVENTS_PER_DATAPOINT = 6

    delays_sorted = delays_sorted[
        delays_sorted["delay_seq"] <= MAX_DELAY_EVENTS_PER_DATAPOINT
    ]

    delay_wide = delays_sorted.pivot(
        index="datapoint_id",
        columns="delay_seq",
        values=[
            "delay_code",
            "delay_code_group",
            "delay_minutes",
            "included_by_cn_filter",
        ]
    )

    delay_wide.columns = [
        f"{field}_{seq}"
        for field, seq in delay_wide.columns
    ]

    delay_wide = delay_wide.reset_index()

    # =====================================================
    # FINAL TABLE
    # =====================================================
    print("Creating final dataframe...")

    final = base.merge(
        delay_wide,
        on="datapoint_id",
        how="left"
    )

    final["train_name"] = final["train_name"].fillna("")
    final["train_type"] = final["train_type"].fillna("Unknown")
    final["dp_name"] = final["dp_name"].fillna("")

    final["train_label"] = (
        final["generated_train_id"]
        .astype("Int64")
        .astype(str)
        + " | "
        + final["train_name"]
    )

    # =====================================================
    # ADD TOTAL DELAY COLUMNS
    # =====================================================
    delay_min_cols = [
        c for c in final.columns
        if c.startswith("delay_minutes_")
    ]

    final["total_delay_min_all_codes"] = (
        final[delay_min_cols]
        .sum(axis=1, skipna=True)
        if delay_min_cols
        else 0
    )

    cn_filtered_delay_parts = []

    for col in delay_min_cols:
        seq = col.split("_")[-1]
        filter_col = f"included_by_cn_filter_{seq}"

        if filter_col in final.columns:
            cn_filtered_delay_parts.append(
                final[col].where(final[filter_col] == True, 0)
            )

    if cn_filtered_delay_parts:
        final["total_delay_min_cn_filtered"] = (
            pd.concat(cn_filtered_delay_parts, axis=1)
            .sum(axis=1, skipna=True)
        )
    else:
        final["total_delay_min_cn_filtered"] = 0

    # =====================================================
    # FINAL COLUMN ORDER
    # =====================================================
    core_cols = [
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
        c for c in final.columns
        if c.startswith("delay_code_")
        or c.startswith("delay_code_group_")
        or c.startswith("delay_minutes_")
        or c.startswith("included_by_cn_filter_")
    ]

    final = final[
        [c for c in core_cols if c in final.columns]
        + sorted(delay_cols, key=lambda x: (x.rsplit("_", 1)[-1], x))
    ]

    # =====================================================
    # OPTIMIZE TYPES
    # =====================================================
    print("Optimizing data types...")

    final = optimize_types(final)

    # =====================================================
    # SAVE OUTPUTS
    # =====================================================
    out_base = RESULTS_PATH / output_json_path.stem

    parquet_path = out_base.with_suffix(".parquet")
    csv_sample_path = out_base.with_suffix(".sample.csv")

    final.to_parquet(
        parquet_path,
        index=False,
        compression="brotli"
    )

    # Only save sample CSV, not full CSV
    final.to_csv(
        csv_sample_path,
        index=False
    )

    print("Done.")
    print(f"Saved parquet: {parquet_path}")
    print(f"Saved sample CSV: {csv_sample_path}")
    print(f"Rows: {len(final):,}")
    print(f"Columns: {len(final.columns):,}")
    print(f"Memory MB: {final.memory_usage(deep=True).sum() / 1024 / 1024:.1f}")


# =====================================================
# RUN
# =====================================================
if __name__ == "__main__":
    process_file(
        OUTPUT_FILE,
        DP_FILE
    )