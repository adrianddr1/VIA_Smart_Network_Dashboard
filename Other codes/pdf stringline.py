# -*- coding: utf-8 -*-
"""
Created on Thu May 28 11:27:35 2026

@author: ZhaoJ
"""
from pathlib import Path
import math
import re

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import transforms


# =====================================================
# USER INPUTS
# =====================================================

PARQUET_FILE = Path(
    r"C:\Users\ZhaoJ\OneDrive - DB E.C.O. North America\Desktop\VIA Dashboards\Inputs\Processed json\TOR_MNT_base_Plant_+6VIA-input-UC-1-output-RT_F0_P0_.parquet"
)

CONFIG_FILE = Path(
    r"C:\Users\ZhaoJ\OneDrive - DB E.C.O. North America\Desktop\VIA Dashboards\Inputs\Other info TOR_MNT_base_Plant_+6VIA.xlsx"
)

PLANT_SHEET_NAME = "Plant Configuration"

SCHEDULE_SHEETS = [
    "Scheduled Trains P",
    "Scheduled Trains Others",
]

OUTPUT_PDF = Path(
    r"C:\Users\ZhaoJ\OneDrive - DB E.C.O. North America\Desktop\VIA Dashboards\+6VIA_Stringlines_24hr_with_weekly_TPC.pdf"
)

WINDOW_HOURS = 24

MAX_TRAINS_PER_PAGE = None
MAX_Y_LABELS = 55
MAX_SCHEDULED_TPC_TRAINS_PER_DAY = None

# Label logic:
# If the page has more trains than this, label passenger trains only.
# If the page has this many or fewer trains, label all trains.
LABEL_ALL_TRAINS_WHEN_TRAIN_COUNT_LESS_THAN = 60

MAX_LABELS_PER_PAGE = 100


# =====================================================
# FIXED COLOR MAP
# =====================================================

FIXED_TRAIN_TYPE_COLORS = {
    "P": "#d62728",   # red - passenger
    "Z": "#1f77b4",
    "L": "#2ca02c",
    "M": "#9467bd",
    "Q": "#ff7f0e",
    "B": "#17becf",
    "A": "#8c564b",
    "X": "#e377c2",
    "G": "#7f7f7f",
    "C": "#bcbd22",
    "S": "#003f5c",
    "T": "#ffa600",
    "E": "#3366cc",
    "Unknown": "#000000",
}

FALLBACK_COLORS = [
    "#3366cc",
    "#109618",
    "#990099",
    "#0099c6",
    "#dd4477",
    "#66aa00",
    "#b82e2e",
    "#316395",
    "#994499",
    "#22aa99",
]


# =====================================================
# BASIC HELPERS
# =====================================================

def normalize_col_name(c):
    c = str(c).strip().upper()
    c = re.sub(r"\s+", " ", c)
    return c


def normalize_station_name(x):
    if pd.isna(x):
        return ""

    x = str(x).strip().upper()
    x = x.replace(" ", "_")
    x = re.sub(r"[^A-Z0-9_]", "", x)
    x = re.sub(r"_+", "_", x)

    return x


def get_train_type_code(train_name):
    if pd.isna(train_name):
        return "Unknown"

    train_name = str(train_name).strip()

    if train_name == "":
        return "Unknown"

    return train_name[0].upper()


def extract_base_train_name_from_parquet(parquet_train_name):
    """
    Matching rule:
        parquet train_name before first dash == scheduled TRAIN NAME

    Example:
        P033A-sched-P42_5HEP_WB-west -> P033A
        A401-sched-65_11K0.8_WB-west -> A401
    """
    p_name = str(parquet_train_name).strip()

    if "-" in p_name:
        return p_name.split("-")[0].strip()

    return p_name


def build_global_color_map(train_type_codes):
    color_map = {}
    fallback_i = 0

    for code in sorted(train_type_codes):
        code = str(code)

        if code in FIXED_TRAIN_TYPE_COLORS:
            color_map[code] = FIXED_TRAIN_TYPE_COLORS[code]
        else:
            color_map[code] = FALLBACK_COLORS[fallback_i % len(FALLBACK_COLORS)]
            fallback_i += 1

    if "P" in train_type_codes:
        color_map["P"] = "#d62728"

    return color_map


def parse_days_of_week(value):
    """
    DAYS OF WEEK examples:
        1234567
        12345
        67
        0

    Assumption:
        1-7 = operating days
        0 or blank = all days
    """
    if pd.isna(value):
        return set(range(1, 8))

    text = str(value).strip()

    if text == "" or text.lower() == "nan":
        return set(range(1, 8))

    if text.endswith(".0"):
        text = text[:-2]

    if text == "0":
        return set(range(1, 8))

    days = set()

    for ch in text:
        if ch in "1234567":
            days.add(int(ch))

    if not days:
        return set(range(1, 8))

    return days


def clean_speed_class_token(x):
    if pd.isna(x):
        return ""

    x = str(x).strip()
    x = x.replace(" ", "")
    x = x.replace("-", "_")

    return x


def scheduled_dep_time_to_hours(value):
    """
    Handles:
    - decimal hour: 21.89
    - Excel time fraction: 0.38
    - string time: 9:00
    - pandas Timestamp / datetime time
    """
    if pd.isna(value):
        return None

    if hasattr(value, "hour") and hasattr(value, "minute"):
        try:
            return value.hour + value.minute / 60 + value.second / 3600
        except Exception:
            pass

    if isinstance(value, str):
        v = value.strip()

        if v == "":
            return None

        if ":" in v:
            parts = v.split(":")
            try:
                h = float(parts[0])
                m = float(parts[1]) if len(parts) > 1 else 0
                s = float(parts[2]) if len(parts) > 2 else 0
                return h + m / 60 + s / 3600
            except Exception:
                return None

        try:
            value = float(v)
        except Exception:
            return None

    try:
        value = float(value)
    except Exception:
        return None

    # Excel time fraction: 0.38 means about 9:07
    if 0 <= value < 1:
        return value * 24

    return value


def make_plant_axis_label(row):
    return str(row.get("plant_name", "")).strip()


# =====================================================
# PLANT CONFIGURATION
# =====================================================

def find_header_row_for_plant(excel_path, sheet_name):
    preview = pd.read_excel(
        excel_path,
        sheet_name=sheet_name,
        header=None,
        nrows=30,
        engine="openpyxl",
    )

    for i in range(len(preview)):
        values = [
            str(x).strip().upper()
            for x in preview.iloc[i].tolist()
            if pd.notna(x)
        ]

        if "MP" in values and "NAME" in values:
            return i

    raise ValueError("Could not find Plant Configuration header row with MP and NAME.")


def read_plant_config(config_file, sheet_name):
    header_row = find_header_row_for_plant(config_file, sheet_name)

    plant = pd.read_excel(
        config_file,
        sheet_name=sheet_name,
        header=header_row,
        engine="openpyxl",
    )

    plant.columns = [str(c).strip() for c in plant.columns]

    rename_map = {}

    for c in plant.columns:
        c_clean = normalize_col_name(c)

        if c_clean == "MP":
            rename_map[c] = "plant_mp"
        elif c_clean == "NAME":
            rename_map[c] = "plant_name"
        elif c_clean == "SWITCH TYPE":
            rename_map[c] = "switch_type"
        elif c_clean == "TRACKS":
            rename_map[c] = "tracks"
        elif c_clean == "SIDING TRACKS":
            rename_map[c] = "siding_tracks"
        elif c_clean in ["# ESTBND", "#ESTBND", "ESTBND"]:
            rename_map[c] = "estbnd"
        elif c_clean in ["#WSTBND", "# WSTBND", "WSTBND"]:
            rename_map[c] = "wstbnd"
        elif c_clean == "TURNOUT":
            rename_map[c] = "turnout"
        elif c_clean == "TURNOUT TRACK":
            rename_map[c] = "turnout_track"

    plant = plant.rename(columns=rename_map)

    required = ["plant_mp", "plant_name"]
    missing = [c for c in required if c not in plant.columns]

    if missing:
        raise ValueError(f"Plant Configuration missing required columns: {missing}")

    optional_cols = [
        "switch_type",
        "tracks",
        "siding_tracks",
        "estbnd",
        "wstbnd",
        "turnout",
        "turnout_track",
    ]

    for col in optional_cols:
        if col not in plant.columns:
            plant[col] = pd.NA

    plant = plant[
        [
            "plant_mp",
            "plant_name",
            "switch_type",
            "tracks",
            "siding_tracks",
            "estbnd",
            "wstbnd",
            "turnout",
            "turnout_track",
        ]
    ].copy()

    plant["plant_mp"] = pd.to_numeric(plant["plant_mp"], errors="coerce")
    plant["tracks"] = pd.to_numeric(plant["tracks"], errors="coerce")
    plant["siding_tracks"] = pd.to_numeric(plant["siding_tracks"], errors="coerce")

    plant["plant_name"] = plant["plant_name"].astype(str).str.strip()
    plant["switch_type"] = plant["switch_type"].astype(str).str.strip()

    plant = plant[
        plant["plant_mp"].notna()
        & plant["plant_name"].notna()
        & (plant["plant_name"] != "")
        & (plant["plant_name"].str.lower() != "nan")
    ].copy()

    plant = plant.drop_duplicates(subset=["plant_mp", "plant_name"])
    plant = plant.sort_values("plant_mp").reset_index(drop=True)

    plant["has_siding"] = (
        plant["siding_tracks"].notna()
        & (plant["siding_tracks"] > 0)
    )

    def siding_number_text(x):
        if pd.isna(x) or float(x) <= 0:
            return ""
        return str(int(x)) if float(x).is_integer() else str(x)

    plant["siding_label"] = plant["siding_tracks"].apply(siding_number_text)
    plant["axis_label"] = plant.apply(make_plant_axis_label, axis=1)

    print(f"Plant rows loaded: {len(plant):,}")
    print(f"Plant MP range: {plant['plant_mp'].min()} to {plant['plant_mp'].max()}")

    return plant


# =====================================================
# SCHEDULED TRAINS
# =====================================================

def find_header_row_for_schedule(excel_path, sheet_name):
    preview = pd.read_excel(
        excel_path,
        sheet_name=sheet_name,
        header=None,
        nrows=80,
        engine="openpyxl",
    )

    for i in range(len(preview)):
        row_vals = [
            normalize_col_name(x)
            for x in preview.iloc[i].tolist()
            if pd.notna(x)
        ]

        has_train = any("TRAIN" in x for x in row_vals)
        has_dep = any(
            (
                ("DEP" in x and "TIME" in x)
                or ("DEPART" in x and "TIME" in x)
                or ("START" in x and "TIME" in x)
            )
            for x in row_vals
        )

        if has_train and has_dep:
            return i

    raise ValueError(f"Could not find header row in schedule sheet: {sheet_name}")


def classify_schedule_column(col_name):
    c = normalize_col_name(col_name)

    if "SPEED" in c and "CLASS" in c:
        return "speed_class"

    if (
        ("DEP" in c and "TIME" in c)
        or ("DEPART" in c and "TIME" in c)
        or ("START" in c and "TIME" in c)
    ):
        if "EARLY" not in c and "LATE" not in c and "S.D" not in c and "SD" not in c:
            return "dep_time"

    if "TRAIN" in c:
        return "scheduled_train_name"

    if "DAYS" in c:
        return "days_of_week"

    if "FIRST" in c and "SWITCH" in c:
        return "first_switch"

    if "LAST" in c and "SWITCH" in c:
        return "last_switch"

    if "PERCENT" in c:
        return "percent"

    if "EARLY" in c:
        return "sd_early"

    if "LATE" in c:
        return "sd_late"

    return None


def read_one_schedule_sheet(config_file, sheet_name):
    header_row = find_header_row_for_schedule(config_file, sheet_name)

    sched_raw = pd.read_excel(
        config_file,
        sheet_name=sheet_name,
        header=header_row,
        engine="openpyxl",
    )

    sched_raw.columns = [str(c).strip() for c in sched_raw.columns]

    print(f"\n--- Schedule sheet: {sheet_name} ---")
    print(f"Header row used: {header_row}")
    print("Raw columns:")
    for c in sched_raw.columns:
        print(f"  {c}")

    rename_map = {}

    for c in sched_raw.columns:
        mapped = classify_schedule_column(c)
        if mapped is not None and mapped not in rename_map.values():
            rename_map[c] = mapped

    print("Column mapping:")
    for old, new in rename_map.items():
        print(f"  {old} -> {new}")

    sched = sched_raw.rename(columns=rename_map).copy()

    needed_cols = [
        "scheduled_train_name",
        "speed_class",
        "dep_time",
        "days_of_week",
        "first_switch",
        "last_switch",
        "percent",
        "sd_early",
        "sd_late",
    ]

    for col in needed_cols:
        if col not in sched.columns:
            sched[col] = pd.NA

    sched = sched[needed_cols].copy()
    sched["source_sheet"] = sheet_name

    print(f"Rows before cleaning: {len(sched):,}")

    sched["scheduled_train_name"] = sched["scheduled_train_name"].astype(str).str.strip()
    sched["speed_class"] = sched["speed_class"].astype(str).str.strip()
    sched["first_switch"] = sched["first_switch"].astype(str).str.strip()
    sched["last_switch"] = sched["last_switch"].astype(str).str.strip()

    sched["scheduled_dep_hour"] = sched["dep_time"].apply(scheduled_dep_time_to_hours)

    debug_nonblank_train = (
        sched["scheduled_train_name"].notna()
        & (sched["scheduled_train_name"] != "")
        & (sched["scheduled_train_name"].str.lower() != "nan")
    ).sum()

    debug_nonblank_dep = sched["dep_time"].notna().sum()
    debug_valid_dep = sched["scheduled_dep_hour"].notna().sum()

    print(f"Rows with nonblank train name: {debug_nonblank_train:,}")
    print(f"Rows with nonblank dep_time raw: {debug_nonblank_dep:,}")
    print(f"Rows with valid scheduled_dep_hour: {debug_valid_dep:,}")

    sched = sched[
        sched["scheduled_train_name"].notna()
        & (sched["scheduled_train_name"] != "")
        & (sched["scheduled_train_name"].str.lower() != "nan")
        & sched["scheduled_dep_hour"].notna()
    ].copy()

    sched = sched[
        ~sched["scheduled_train_name"].str.upper().isin(
            [
                "TRAINS#",
                "TRAIN#",
                "TRAIN",
                "TRAINS",
                "TRAIN NAME",
                "MONTREAL",
                "OTTAWA",
                "ROAD",
            ]
        )
    ].copy()

    # Keep only rows that look like train symbols.
    sched = sched[
        sched["scheduled_train_name"].str.match(r"^[A-Za-z]+\d+", na=False)
    ].copy()

    sched["operating_days"] = sched["days_of_week"].apply(parse_days_of_week)
    sched["scheduled_train_type_code"] = sched["scheduled_train_name"].apply(get_train_type_code)
    sched["speed_class_token"] = sched["speed_class"].apply(clean_speed_class_token)
    sched["scheduled_train_name_norm"] = sched["scheduled_train_name"].apply(normalize_station_name)
    sched["first_switch_norm"] = sched["first_switch"].apply(normalize_station_name)
    sched["last_switch_norm"] = sched["last_switch"].apply(normalize_station_name)

    print(f"Rows after cleaning: {len(sched):,}")

    if len(sched) > 0:
        print("Sample cleaned rows:")
        print(
            sched[
                [
                    "scheduled_train_name",
                    "speed_class",
                    "scheduled_dep_hour",
                    "days_of_week",
                    "first_switch",
                    "last_switch",
                    "source_sheet",
                ]
            ].head(15)
        )
    else:
        print("WARNING: This sheet produced zero usable scheduled train rows.")

    return sched


def read_scheduled_trains(config_file, sheet_names):
    frames = []

    for sheet_name in sheet_names:
        try:
            one = read_one_schedule_sheet(config_file, sheet_name)
            if not one.empty:
                frames.append(one)
            print(f"Scheduled trains loaded from {sheet_name}: {len(one):,}")
        except Exception as e:
            print(f"WARNING: Could not read schedule sheet {sheet_name}: {e}")

    if not frames:
        return pd.DataFrame()

    sched = pd.concat(frames, ignore_index=True)

    sched = sched.sort_values(
        ["scheduled_dep_hour", "scheduled_train_name"]
    ).reset_index(drop=True)

    print(f"Total scheduled train rows loaded: {len(sched):,}")

    print("Scheduled trains by source sheet:")
    print(sched.groupby("source_sheet").size())

    passenger_debug = sched[
        sched["scheduled_train_name"].astype(str).str.startswith("P")
    ][
        [
            "scheduled_train_name",
            "speed_class",
            "scheduled_dep_hour",
            "days_of_week",
            "first_switch",
            "last_switch",
            "source_sheet",
        ]
    ].head(40)


    return sched


def match_parquet_train_name_to_schedule(parquet_train_name, scheduled):
    """
    Exact intended rule:
        parquet train_name before first dash == scheduled TRAIN NAME
    """
    p_name = str(parquet_train_name).strip()

    if p_name == "" or p_name.lower() == "nan":
        return None

    parquet_base_name = extract_base_train_name_from_parquet(p_name)

    matches = []

    for idx, row in scheduled.iterrows():
        sched_train = str(row["scheduled_train_name"]).strip()

        if sched_train == "" or sched_train.lower() == "nan":
            continue

        if parquet_base_name != sched_train:
            continue

        score = 1000

        if parquet_base_name.startswith("P") and row.get("source_sheet", "") == "Scheduled Trains P":
            score += 100

        if not parquet_base_name.startswith("P") and row.get("source_sheet", "") == "Scheduled Trains Others":
            score += 100

        speed_token = str(row.get("speed_class_token", "")).strip()

        comparable_p = (
            p_name
            .replace("-", "_")
            .replace(" ", "")
            .upper()
        )

        comparable_speed = (
            speed_token
            .replace("-", "_")
            .replace(" ", "")
            .upper()
        )

        if comparable_speed and comparable_speed not in ["NAN", "<NA>"]:
            if comparable_speed in comparable_p:
                score += 50

        matches.append((score, idx))

    if not matches:
        return None

    matches = sorted(matches, reverse=True)
    best_idx = matches[0][1]

    return scheduled.loc[best_idx]


# =====================================================
# PARQUET STRINGLINE DATA
# =====================================================

def clean_stringline_data(df):
    required_cols = [
        "train_label",
        "train_name",
        "dp_name",
        "mileage",
        "arrival_seconds",
        "departure_seconds",
    ]

    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns in parquet: {missing}")

    df = df.copy()

    df["train_label"] = df["train_label"].astype(str)
    df["train_name"] = df["train_name"].astype(str)
    df["dp_name"] = df["dp_name"].astype(str)

    df["mileage"] = pd.to_numeric(df["mileage"], errors="coerce").astype("float64")
    df["arrival_seconds"] = pd.to_numeric(df["arrival_seconds"], errors="coerce")
    df["departure_seconds"] = pd.to_numeric(df["departure_seconds"], errors="coerce")

    bad_mask = (
        (df["departure_seconds"] == 0)
        & (df["arrival_seconds"] > 0)
    )

    print(f"Rows before cleaning: {len(df):,}")
    print(f"Invalid / never-dispatched rows removed: {bad_mask.sum():,}")

    df = df[~bad_mask].copy()

    df = df[
        df["mileage"].notna()
        & df["arrival_seconds"].notna()
        & df["departure_seconds"].notna()
        & (df["departure_seconds"] >= df["arrival_seconds"])
    ].copy()

    df["arrival_hour"] = df["arrival_seconds"] / 3600
    df["departure_hour"] = df["departure_seconds"] / 3600

    df["train_type_code"] = df["train_name"].apply(get_train_type_code)

    return df


def assign_axis_labels_from_plant(df, plant):
    """
    Match each stringline record to nearest Plant Configuration MP.
    Fixes merge_asof dtype error by forcing both MP fields to float64.
    """
    df = df.copy()
    plant = plant.copy()

    # Force exact same dtype
    df["mileage"] = pd.to_numeric(df["mileage"], errors="coerce").astype("float64")
    plant["plant_mp"] = pd.to_numeric(plant["plant_mp"], errors="coerce").astype("float64")

    plant_axis = plant[
        [
            "plant_mp",
            "plant_name",
            "axis_label",
            "has_siding",
            "siding_label",
        ]
    ].copy()

    plant_axis = plant_axis[plant_axis["plant_mp"].notna()].copy()

    df_with_mileage = df[df["mileage"].notna()].copy()
    df_no_mileage = df[df["mileage"].isna()].copy()

    plant_axis = plant_axis.sort_values("plant_mp").reset_index(drop=True)
    df_with_mileage = df_with_mileage.sort_values("mileage").reset_index(drop=False)

    matched = pd.merge_asof(
        df_with_mileage,
        plant_axis,
        left_on="mileage",
        right_on="plant_mp",
        direction="nearest",
        tolerance=0.05,
    )

    matched["axis_label"] = matched["axis_label"].fillna(matched["dp_name"])
    matched["has_siding"] = matched["has_siding"].fillna(False)
    matched["siding_label"] = matched["siding_label"].fillna("")

    matched = matched.set_index("index").sort_index()

    if not df_no_mileage.empty:
        df_no_mileage["plant_mp"] = pd.NA
        df_no_mileage["plant_name"] = pd.NA
        df_no_mileage["axis_label"] = df_no_mileage["dp_name"]
        df_no_mileage["has_siding"] = False
        df_no_mileage["siding_label"] = ""

        matched = pd.concat([matched, df_no_mileage], axis=0).sort_index()

    return matched


# =====================================================
# TPC CALCULATION
# =====================================================

def build_tpc_profile_for_train_name(data, selected_train_name):
    tn = data[
        (data["train_name"].astype(str) == str(selected_train_name))
        & (data["arrival_seconds"].notna())
        & (data["departure_seconds"].notna())
        & (data["mileage"].notna())
    ].copy()

    tn = tn[
        (tn["departure_seconds"] >= tn["arrival_seconds"])
        & (tn["departure_seconds"] > 0)
    ].copy()

    if tn.empty:
        return pd.DataFrame(), None, None

    route_label = (
        tn.groupby("train_label")
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
        return pd.DataFrame(), route_label, None

    first_route_dp_name = str(route.iloc[0]["dp_name"])

    if "dwell_minutes" in tn.columns:
        min_dwell_by_dp = (
            tn[
                tn["dwell_minutes"].notna()
                & (tn["dwell_minutes"] >= 0)
            ]
            .groupby("dp_id")["dwell_minutes"]
            .min()
            .to_dict()
        )
    else:
        min_dwell_by_dp = {}

    tn = tn.sort_values(["train_label", "arrival_seconds"]).copy()

    tn["next_dp_id"] = tn.groupby("train_label")["dp_id"].shift(-1)
    tn["next_arrival_seconds"] = tn.groupby("train_label")["arrival_seconds"].shift(-1)
    tn["link_seconds"] = tn["next_arrival_seconds"] - tn["departure_seconds"]

    seg = tn[
        tn["next_dp_id"].notna()
        & tn["link_seconds"].notna()
        & (tn["link_seconds"] >= 0)
    ].copy()

    seg["next_dp_id"] = seg["next_dp_id"].astype(int)

    min_link_by_pair = (
        seg.groupby(["dp_id", "next_dp_id"])["link_seconds"]
        .min()
        .to_dict()
    )

    rows = []
    current_time = 0.0

    for i in range(len(route)):
        r = route.iloc[i]
        dp_id = int(r["dp_id"])

        dwell_min = float(min_dwell_by_dp.get(dp_id, 0.0))
        dwell_sec = dwell_min * 60.0

        arrival_t = current_time
        departure_t = arrival_t + dwell_sec

        rows.append(
            {
                "tpc_seconds": arrival_t,
                "tpc_hours": arrival_t / 3600,
                "mileage": float(r["mileage"]),
                "dp_id": dp_id,
                "dp_name": str(r["dp_name"]),
                "event": "arrival",
                "tpc_dwell_minutes": dwell_min,
            }
        )

        rows.append(
            {
                "tpc_seconds": departure_t,
                "tpc_hours": departure_t / 3600,
                "mileage": float(r["mileage"]),
                "dp_id": dp_id,
                "dp_name": str(r["dp_name"]),
                "event": "departure",
                "tpc_dwell_minutes": dwell_min,
            }
        )

        if i < len(route) - 1:
            next_r = route.iloc[i + 1]
            next_dp_id = int(next_r["dp_id"])

            link_sec = min_link_by_pair.get((dp_id, next_dp_id), None)

            if link_sec is None:
                link_sec = float(next_r["arrival_seconds"] - r["departure_seconds"])

            if pd.isna(link_sec) or link_sec < 0:
                link_sec = 0.0

            current_time = departure_t + link_sec

    tpc = pd.DataFrame(rows)

    return tpc, route_label, first_route_dp_name


def slice_tpc_profile_between_switches(tpc_profile, first_switch, last_switch):
    if tpc_profile.empty:
        return pd.DataFrame(), "Empty TPC profile"

    tpc = tpc_profile.copy()
    tpc["dp_norm"] = tpc["dp_name"].apply(normalize_station_name)

    first_norm = normalize_station_name(first_switch)
    last_norm = normalize_station_name(last_switch)

    if first_norm in ["", "NAN"]:
        first_idx = tpc.index.min()
    else:
        first_matches = tpc[tpc["dp_norm"] == first_norm]

        if first_matches.empty:
            return pd.DataFrame(), f"FIRST SWITCH not found in TPC route: {first_switch}"

        first_idx = first_matches.index.min()

    if last_norm in ["", "NAN"]:
        last_idx = tpc.index.max()
    else:
        last_matches = tpc[tpc["dp_norm"] == last_norm]

        if last_matches.empty:
            return pd.DataFrame(), f"LAST SWITCH not found in TPC route: {last_switch}"

        last_idx = last_matches.index.max()

    if last_idx <= first_idx:
        return pd.DataFrame(), f"LAST SWITCH occurs before FIRST SWITCH: {first_switch} -> {last_switch}"

    sliced = tpc.loc[first_idx:last_idx].copy()

    if sliced.empty:
        return pd.DataFrame(), "TPC slice is empty"

    t0 = sliced["tpc_seconds"].min()

    sliced["tpc_seconds"] = sliced["tpc_seconds"] - t0
    sliced["tpc_hours"] = sliced["tpc_seconds"] / 3600

    sliced = sliced.drop(columns=["dp_norm"], errors="ignore")

    return sliced, "OK"


def build_weekly_scheduled_tpc_runs(df, scheduled):
    if scheduled.empty:
        return pd.DataFrame(), pd.DataFrame()

    parquet_train_names = sorted(df["train_name"].astype(str).dropna().unique())

    tpc_cache = {}
    plotted_rows = []
    match_rows = []

    for parquet_train_name in parquet_train_names:
        sched_row = match_parquet_train_name_to_schedule(
            parquet_train_name,
            scheduled,
        )

        parquet_base_train_name = extract_base_train_name_from_parquet(parquet_train_name)

        if sched_row is None:
            match_rows.append(
                {
                    "parquet_train_name": parquet_train_name,
                    "parquet_base_train_name": parquet_base_train_name,
                    "scheduled_train_name": None,
                    "speed_class": None,
                    "source_sheet": None,
                    "status": "No schedule match",
                }
            )
            continue

        scheduled_train_name = sched_row["scheduled_train_name"]
        speed_class = sched_row["speed_class"]
        scheduled_dep_hour = sched_row["scheduled_dep_hour"]
        operating_days = sched_row["operating_days"]
        first_switch = sched_row["first_switch"]
        last_switch = sched_row["last_switch"]

        if parquet_train_name not in tpc_cache:
            full_tpc_profile, route_label, first_route_dp_name = build_tpc_profile_for_train_name(
                df,
                parquet_train_name,
            )

            tpc_cache[parquet_train_name] = (
                full_tpc_profile,
                route_label,
                first_route_dp_name,
            )

        full_tpc_profile, route_label, first_route_dp_name = tpc_cache[parquet_train_name]

        if full_tpc_profile.empty:
            match_rows.append(
                {
                    "parquet_train_name": parquet_train_name,
                    "parquet_base_train_name": parquet_base_train_name,
                    "scheduled_train_name": scheduled_train_name,
                    "speed_class": speed_class,
                    "source_sheet": sched_row["source_sheet"],
                    "first_switch": first_switch,
                    "last_switch": last_switch,
                    "status": "Matched schedule but no TPC profile",
                }
            )
            continue

        tpc_profile, slice_status = slice_tpc_profile_between_switches(
            full_tpc_profile,
            first_switch,
            last_switch,
        )

        if tpc_profile.empty:
            match_rows.append(
                {
                    "parquet_train_name": parquet_train_name,
                    "parquet_base_train_name": parquet_base_train_name,
                    "scheduled_train_name": scheduled_train_name,
                    "speed_class": speed_class,
                    "source_sheet": sched_row["source_sheet"],
                    "scheduled_dep_hour": scheduled_dep_hour,
                    "operating_days": "".join(str(d) for d in sorted(operating_days)),
                    "first_switch": first_switch,
                    "last_switch": last_switch,
                    "first_tpc_dp_name": first_route_dp_name,
                    "route_template_train_label": route_label,
                    "status": slice_status,
                }
            )
            continue

        first_plotted_dp_name = str(
            tpc_profile.sort_values("tpc_seconds")["dp_name"].iloc[0]
        )

        last_plotted_dp_name = str(
            tpc_profile.sort_values("tpc_seconds")["dp_name"].iloc[-1]
        )

        for day in range(1, 8):
            if day not in operating_days:
                continue

            shifted = tpc_profile.copy()

            shifted["week_day"] = day
            shifted["scheduled_train_name"] = scheduled_train_name
            shifted["speed_class"] = speed_class
            shifted["parquet_train_name"] = parquet_train_name
            shifted["parquet_base_train_name"] = parquet_base_train_name
            shifted["source_sheet"] = sched_row["source_sheet"]
            shifted["scheduled_dep_hour"] = scheduled_dep_hour
            shifted["route_template_train_label"] = route_label
            shifted["train_type_code"] = get_train_type_code(scheduled_train_name)
            shifted["first_switch"] = first_switch
            shifted["last_switch"] = last_switch
            shifted["first_plotted_dp_name"] = first_plotted_dp_name
            shifted["last_plotted_dp_name"] = last_plotted_dp_name
            shifted["plot_hour"] = scheduled_dep_hour + shifted["tpc_hours"]

            plotted_rows.append(shifted)

        match_rows.append(
            {
                "parquet_train_name": parquet_train_name,
                "parquet_base_train_name": parquet_base_train_name,
                "scheduled_train_name": scheduled_train_name,
                "speed_class": speed_class,
                "source_sheet": sched_row["source_sheet"],
                "scheduled_dep_hour": scheduled_dep_hour,
                "operating_days": "".join(str(d) for d in sorted(operating_days)),
                "first_switch": first_switch,
                "last_switch": last_switch,
                "first_plotted_dp_name": first_plotted_dp_name,
                "last_plotted_dp_name": last_plotted_dp_name,
                "route_template_train_label": route_label,
                "tpc_duration_hours": tpc_profile["tpc_hours"].max(),
                "status": "Matched and plotted",
            }
        )

    if plotted_rows:
        plot_df = pd.concat(plotted_rows, ignore_index=True)
    else:
        plot_df = pd.DataFrame()

    match_df = pd.DataFrame(match_rows)

    return plot_df, match_df


# =====================================================
# AXIS LABELS, HIGHLIGHTS, AND TRAIN LABELS
# =====================================================

def build_dp_axis_label_table(df_window, plant, max_labels=55):
    y_min = df_window["mileage"].min()
    y_max = df_window["mileage"].max()

    if pd.isna(y_min) or pd.isna(y_max):
        return pd.DataFrame()

    plant_slice = plant[
        (plant["plant_mp"] >= y_min)
        & (plant["plant_mp"] <= y_max)
    ].copy()

    if not plant_slice.empty:
        label_df = plant_slice[
            [
                "plant_mp",
                "axis_label",
                "has_siding",
                "siding_label",
            ]
        ].rename(columns={"plant_mp": "mileage"})
    else:
        label_df = (
            df_window[
                [
                    "mileage",
                    "axis_label",
                    "has_siding",
                    "siding_label",
                ]
            ]
            .dropna(subset=["mileage", "axis_label"])
            .drop_duplicates()
            .sort_values("mileage")
        )

    if label_df.empty:
        return label_df

    if len(label_df) > max_labels:
        step = math.ceil(len(label_df) / max_labels)
        label_df = label_df.iloc[::step].copy()

    return label_df.sort_values("mileage")


def draw_custom_y_axis_labels(ax, label_df):
    ax.set_yticks(label_df["mileage"].tolist())
    ax.set_yticklabels([])

    trans = transforms.blended_transform_factory(
        ax.transAxes,
        ax.transData
    )

    for _, row in label_df.iterrows():
        y = row["mileage"]
        name = str(row["axis_label"])
        has_siding = bool(row.get("has_siding", False))
        siding_label = str(row.get("siding_label", "")).strip()

        if has_siding and siding_label:
            ax.text(
                -0.012,
                y,
                name,
                transform=trans,
                ha="right",
                va="center",
                fontsize=6,
                fontweight="bold",
                color="black",
                clip_on=False,
            )

            ax.text(
                -0.010,
                y,
                f" ({siding_label})",
                transform=trans,
                ha="left",
                va="center",
                fontsize=6,
                fontweight="bold",
                color="red",
                clip_on=False,
            )

        else:
            ax.text(
                -0.012,
                y,
                name,
                transform=trans,
                ha="right",
                va="center",
                fontsize=6,
                color="black",
                clip_on=False,
            )


def draw_single_track_highlights(ax, plant, y_min, y_max):
    p = plant[
        plant["plant_mp"].notna()
        & plant["tracks"].notna()
    ].copy()

    if p.empty:
        return

    p = p.sort_values("plant_mp").reset_index(drop=True)
    mps = p["plant_mp"].tolist()

    for i, row in p.iterrows():
        tracks = row["tracks"]

        if pd.isna(tracks) or int(tracks) != 1:
            continue

        mp = row["plant_mp"]

        if i == 0:
            lower = mp - 0.5
        else:
            lower = (mps[i - 1] + mp) / 2

        if i == len(p) - 1:
            upper = mp + 0.5
        else:
            upper = (mp + mps[i + 1]) / 2

        if upper < y_min or lower > y_max:
            continue

        ax.axhspan(
            max(lower, y_min),
            min(upper, y_max),
            facecolor="#fff2cc",
            alpha=0.55,
            zorder=0,
        )


def add_common_legend(ax, color_map):
    legend_handles = []

    for code in sorted(color_map.keys()):
        handle = plt.Line2D(
            [0],
            [0],
            color=color_map[code],
            lw=3,
            label=f"{code}"
        )
        legend_handles.append(handle)

    single_track_patch = plt.Line2D(
        [0],
        [0],
        color="#fff2cc",
        lw=8,
        label="Single Track"
    )

    legend_handles.append(single_track_patch)

    ax.legend(
        handles=legend_handles,
        title="Train type\n(first letter)",
        loc="upper right",
        fontsize=8,
        title_fontsize=8,
        frameon=True,
    )


def add_train_end_labels(
    ax,
    plot_df,
    label_col,
    time_col,
    mileage_col,
    only_passenger=True,
    max_labels=100,
):
    """
    Add labels near the end of each line.

    For dense pages, use only_passenger=True.
    For less dense pages, use only_passenger=False.
    """
    if plot_df.empty:
        return

    label_rows = []

    for train_name, g in plot_df.groupby(label_col):
        g = g.sort_values(time_col)

        if g.empty:
            continue

        short_name = str(train_name).split("-")[0]

        if only_passenger and not short_name.startswith("P"):
            continue

        last = g.iloc[-1]

        label_rows.append(
            {
                "label": short_name,
                "x": last[time_col],
                "y": last[mileage_col],
                "train_type_code": str(last.get("train_type_code", "")),
            }
        )

    if not label_rows:
        return

    label_df = pd.DataFrame(label_rows)

    # Sort labels by time so the earlier labels are added first.
    label_df = label_df.sort_values(["x", "y"]).head(max_labels)

    for _, r in label_df.iterrows():
        label = str(r["label"])
        is_passenger = label.startswith("P")

        ax.text(
            r["x"] + 0.05,
            r["y"],
            label,
            fontsize=5.5,
            color="red" if is_passenger else "black",
            va="center",
            ha="left",
            clip_on=True,
            zorder=5,
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                alpha=0.55,
                pad=0.2,
            ),
        )


# =====================================================
# PDF PAGES
# =====================================================

def plot_scheduled_tpc_day_page(tpc_plot_df, match_df, plant, pdf, color_map, day):
    day_df = tpc_plot_df[tpc_plot_df["week_day"] == day].copy()

    if day_df.empty:
        fig, ax = plt.subplots(figsize=(17, 11))
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            f"No scheduled TPC runs for Day {day}.",
            ha="center",
            va="center",
            fontsize=16,
        )
        pdf.savefig(fig)
        plt.close(fig)
        print(f"No scheduled TPC runs for Day {day}.")
        return

    train_order = (
        day_df.groupby("parquet_train_name")["scheduled_dep_hour"]
        .min()
        .sort_values()
        .index
        .tolist()
    )

    if MAX_SCHEDULED_TPC_TRAINS_PER_DAY is not None:
        train_order = train_order[:MAX_SCHEDULED_TPC_TRAINS_PER_DAY]
        plot_df = day_df[
            day_df["parquet_train_name"].isin(train_order)
        ].copy()
    else:
        plot_df = day_df.copy()

    fig, ax = plt.subplots(figsize=(17, 11))

    y_min = plot_df["mileage"].min()
    y_max = plot_df["mileage"].max()

    pad = max((y_max - y_min) * 0.03, 0.5)
    plot_y_min = y_min - pad
    plot_y_max = y_max + pad

    draw_single_track_highlights(ax, plant, plot_y_min, plot_y_max)

    for train_name, g in plot_df.groupby("parquet_train_name"):
        g = g.sort_values("plot_hour")

        code = str(g["train_type_code"].iloc[0])
        color = color_map.get(code, "#000000")

        ax.plot(
            g["plot_hour"].to_numpy(),
            g["mileage"].to_numpy(),
            linewidth=0.95,
            alpha=0.82,
            color=color,
            zorder=2,
        )

    train_count = plot_df["parquet_train_name"].nunique()
    only_passenger_labels = False

    add_train_end_labels(
        ax=ax,
        plot_df=plot_df,
        label_col="scheduled_train_name",
        time_col="plot_hour",
        mileage_col="mileage",
        only_passenger=only_passenger_labels,
        max_labels=MAX_LABELS_PER_PAGE,
    )

    label_df = build_dp_axis_label_table(
        plot_df.rename(columns={"plot_hour": "arrival_hour"}),
        plant,
        max_labels=MAX_Y_LABELS,
    )

    draw_custom_y_axis_labels(ax, label_df)

    ax.set_xlim(0, 24)
    ax.set_ylim(plot_y_min, plot_y_max)

    ax.grid(True, linewidth=0.3, alpha=0.4, zorder=1)

    passenger_count = plot_df[
        plot_df["train_type_code"].astype(str).eq("P")
    ]["parquet_train_name"].nunique()

    other_count = plot_df[
        ~plot_df["train_type_code"].astype(str).eq("P")
    ]["parquet_train_name"].nunique()

    label_note = "passenger labels only" if only_passenger_labels else "all train labels"

    ax.set_title(
        f"Scheduled TPC/Base Runs — Day {day}\n"
        f"Plotted train names: {train_count:,} "
        f"(Passenger: {passenger_count:,}, Other: {other_count:,})",
        fontsize=14,
        fontweight="bold",
    )

    ax.set_xlabel("Scheduled Time of Day + TPC Runtime (hours)")
    ax.set_ylabel("DP Name; red number = siding tracks")

    add_common_legend(ax, color_map)

    footer = (
        f"Day {day} uses DAYS OF WEEK from scheduled train sheets. "
        f"TPC profile uses minimum dwell/link time from parquet. "
        f"TPC is sliced from FIRST SWITCH to LAST SWITCH. "
        f"Generated from: {PARQUET_FILE.name}"
    )

    fig.text(
        0.01,
        0.01,
        footer,
        fontsize=8,
        ha="left",
        va="bottom",
    )

    plt.tight_layout(rect=[0, 0.04, 1, 0.95])

    pdf.savefig(fig)
    plt.close(fig)

    print(
        f"Printed scheduled TPC Day {day}: "
        f"{train_count:,} trains "
        f"(Passenger: {passenger_count:,}, Other: {other_count:,});"
    )


def plot_window(df, plant, start_hour, end_hour, pdf, color_map):
    window = df[
        (df["arrival_hour"] >= start_hour)
        & (df["arrival_hour"] < end_hour)
    ].copy()

    if window.empty:
        print(f"No data for {start_hour:.0f}-{end_hour:.0f} hr, skipping.")
        return

    train_labels = (
        window.groupby("train_label")["arrival_hour"]
        .min()
        .sort_values()
        .index
        .tolist()
    )

    trains_before_cap = len(train_labels)

    if MAX_TRAINS_PER_PAGE is not None:
        train_labels = train_labels[:MAX_TRAINS_PER_PAGE]
        window = window[window["train_label"].isin(train_labels)].copy()

    trains_plotted = window["train_label"].nunique()

    fig, ax = plt.subplots(figsize=(17, 11))

    y_min = window["mileage"].min()
    y_max = window["mileage"].max()

    if pd.notna(y_min) and pd.notna(y_max):
        pad = max((y_max - y_min) * 0.03, 0.5)
        plot_y_min = y_min - pad
        plot_y_max = y_max + pad
    else:
        plot_y_min = None
        plot_y_max = None

    if plot_y_min is not None and plot_y_max is not None:
        draw_single_track_highlights(ax, plant, plot_y_min, plot_y_max)

    for train_label, g in window.groupby("train_label"):
        g = g.sort_values("arrival_seconds")

        if g.empty:
            continue

        train_type_code = str(g["train_type_code"].iloc[0])
        color = color_map.get(train_type_code, "#000000")

        ax.plot(
            g["arrival_hour"].to_numpy(),
            g["mileage"].to_numpy(),
            linewidth=0.85,
            alpha=0.78,
            color=color,
            zorder=2,
        )

    only_passenger_labels = False #trains_plotted > LABEL_ALL_TRAINS_WHEN_TRAIN_COUNT_LESS_THAN

    add_train_end_labels(
        ax=ax,
        plot_df=window,
        label_col="train_name",
        time_col="arrival_hour",
        mileage_col="mileage",
        only_passenger=only_passenger_labels,
        max_labels=MAX_LABELS_PER_PAGE,
    )

    label_df = build_dp_axis_label_table(
        window,
        plant,
        max_labels=MAX_Y_LABELS,
    )

    draw_custom_y_axis_labels(ax, label_df)

    ax.set_xlim(start_hour, end_hour)

    if plot_y_min is not None and plot_y_max is not None:
        ax.set_ylim(plot_y_min, plot_y_max)

    ax.grid(True, linewidth=0.3, alpha=0.4, zorder=1)

    start_day = int(start_hour // 24)
    end_day = int(end_hour // 24)

    passenger_count = window[
        window["train_type_code"].astype(str).eq("P")
    ]["train_label"].nunique()

    other_count = window[
        ~window["train_type_code"].astype(str).eq("P")
    ]["train_label"].nunique()

    label_note = "passenger labels only" if only_passenger_labels else "all train labels"

    ax.set_title(
        f"Stringline: Hour {start_hour:.0f}-{end_hour:.0f} "
        f"(Day {start_day} to {end_day})\n"
        f"Train runs on page: {trains_plotted:,} "
        f"(Passenger: {passenger_count:,}, Other: {other_count:,})",
        fontsize=14,
        fontweight="bold",
    )

    ax.set_xlabel("Simulation Time (hours)")
    ax.set_ylabel("DP Name; red number = siding tracks")

    add_common_legend(ax, color_map)

    footer = (
        f"Window: {start_hour:.0f}-{end_hour:.0f} hr | "
        f"Unique train runs in window before cap: {trains_before_cap:,} | "
        f"Train runs plotted: {trains_plotted:,} | "
        f"Rows plotted: {len(window):,} | "
        f"Generated from: {PARQUET_FILE.name}"
    )

    fig.text(
        0.01,
        0.01,
        footer,
        fontsize=8,
        ha="left",
        va="bottom",
    )

    plt.tight_layout(rect=[0, 0.04, 1, 0.95])

    pdf.savefig(fig)
    plt.close(fig)

    print(
        f"Printed page {start_hour:.0f}-{end_hour:.0f} hr: "
        f"{trains_plotted:,} trains plotted "
        f"(Passenger: {passenger_count:,}, Other: {other_count:,}), "
        f"{len(window):,} rows;"
    )


# =====================================================
# MAIN
# =====================================================

def main():
    print("Reading parquet...")
    print(PARQUET_FILE)

    df = pd.read_parquet(PARQUET_FILE)
    df = clean_stringline_data(df)

    print("Reading Plant Configuration...")
    print(CONFIG_FILE)

    plant = read_plant_config(CONFIG_FILE, PLANT_SHEET_NAME)

    df = assign_axis_labels_from_plant(df, plant)

    print("Reading scheduled train sheets...")
    scheduled = read_scheduled_trains(CONFIG_FILE, SCHEDULE_SHEETS)

    print("Building weekly scheduled TPC runs...")
    tpc_plot_df, match_df = build_weekly_scheduled_tpc_runs(df, scheduled)

    if df.empty:
        raise ValueError("No valid data after cleaning.")

    schedule_codes = (
        set(scheduled["scheduled_train_type_code"].dropna().unique())
        if not scheduled.empty
        else set()
    )

    tpc_codes = (
        set(tpc_plot_df["train_type_code"].dropna().unique())
        if not tpc_plot_df.empty
        else set()
    )

    all_train_type_codes = sorted(
        set(df["train_type_code"].dropna().unique())
        | schedule_codes
        | tpc_codes
    )

    color_map = build_global_color_map(all_train_type_codes)

    max_hour = math.ceil(df["arrival_hour"].max())
    final_window_start = int(math.floor(max_hour / WINDOW_HOURS) * WINDOW_HOURS)

    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)

    print(f"Writing PDF to: {OUTPUT_PDF}")
    print(f"Data max hour: {max_hour:,.1f}")
    print(f"Total train runs: {df['train_label'].nunique():,}")
    print(f"Train type codes: {all_train_type_codes}")


    if not match_df.empty:
        match_csv = OUTPUT_PDF.with_suffix(".weekly_tpc_matches.csv")
        match_df.to_csv(match_csv, index=False)
        print(f"Saved weekly TPC match table: {match_csv}")

        print("TPC match status summary:")
        print(match_df.groupby("status").size())

        if "source_sheet" in match_df.columns:
            print("Matched/plotted by source sheet:")
            print(
                match_df[
                    match_df["status"].eq("Matched and plotted")
                ].groupby("source_sheet").size()
            )

        print("Passenger match status summary:")
        print(
            match_df[
                match_df["parquet_base_train_name"].astype(str).str.startswith("P")
            ].groupby("status").size()
        )

    with PdfPages(OUTPUT_PDF) as pdf:
        # First 7 pages: scheduled TPC/base runs by day of week
        for day in range(1, 8):
            plot_scheduled_tpc_day_page(
                tpc_plot_df,
                match_df,
                plant,
                pdf,
                color_map,
                day,
            )

        # Remaining pages: actual simulation stringlines by 24-hour window
        for start_hour in range(0, final_window_start + 1, WINDOW_HOURS):
            end_hour = start_hour + WINDOW_HOURS
            plot_window(df, plant, start_hour, end_hour, pdf, color_map)

    print("Done.")
    print(OUTPUT_PDF)


if __name__ == "__main__":
    main()