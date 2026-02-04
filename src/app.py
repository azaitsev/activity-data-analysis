from __future__ import annotations

import io
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fitparse import FitFile
from lxml import etree


APP_STATIC_DIR = "src/static"
INDEX_HTML_PATH = f"{APP_STATIC_DIR}/index.html"

SUPPORTED_METRICS = ("hr_bpm", "speed_kmh", "power_w")


app = FastAPI()

app.mount("/static", StaticFiles(directory=APP_STATIC_DIR), name="static")


@app.get("/")
def get_index() -> FileResponse:
    return FileResponse(INDEX_HTML_PATH)


def parse_fit_bytes(file_bytes: bytes) -> pd.DataFrame:
    fit_file = FitFile(io.BytesIO(file_bytes))
    rows: List[dict] = []

    for record in fit_file.get_messages("record"):
        row = {
            "timestamp": None,
            "hr_bpm": None,
            "power_w": None,
            "speed_kmh": None,
        }

        for field in record:
            if field.name == "timestamp":
                row["timestamp"] = field.value
            elif field.name == "heart_rate":
                row["hr_bpm"] = field.value
            elif field.name == "power":
                row["power_w"] = field.value
            elif field.name == "speed":
                if field.value is not None:
                    row["speed_kmh"] = float(field.value) * 3.6

        if row["timestamp"] is not None:
            rows.append(row)

    df_data = pd.DataFrame(rows)
    if df_data.empty:
        return df_data

    df_data["timestamp"] = pd.to_datetime(df_data["timestamp"], utc=True, errors="coerce")
    df_data = df_data.dropna(subset=["timestamp"]).sort_values("timestamp")
    return df_data


def normalize_tcx_nsmap(root: etree._Element) -> Dict[str, str]:
    raw_nsmap = dict(root.nsmap or {})
    normalized_nsmap: Dict[str, str] = {}
    for key, value in raw_nsmap.items():
        if not value:
            continue
        if key is None:
            normalized_nsmap["tcx"] = value
        else:
            normalized_nsmap[key] = value

    if "tcx" not in normalized_nsmap:
        normalized_nsmap["tcx"] = "http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2"

    return normalized_nsmap


def get_first_xpath_text(node: etree._Element, xpath_expr: str, nsmap: Dict[str, str]) -> Optional[str]:
    found = node.xpath(xpath_expr, namespaces=nsmap)
    if not found:
        return None

    first_item = found[0]
    if isinstance(first_item, etree._Element):
        return first_item.text
    return str(first_item)


def safe_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    value_stripped = value.strip()
    if not value_stripped:
        return None
    try:
        return int(float(value_stripped))
    except (ValueError, TypeError):
        return None


def safe_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    value_stripped = value.strip()
    if not value_stripped:
        return None
    try:
        return float(value_stripped)
    except (ValueError, TypeError):
        return None


def parse_tcx_bytes(file_bytes: bytes) -> pd.DataFrame:
    root = etree.fromstring(file_bytes)
    nsmap = normalize_tcx_nsmap(root)

    trackpoints = root.xpath("//tcx:Trackpoint", namespaces=nsmap)
    rows: List[dict] = []

    for trackpoint in trackpoints:
        time_text = get_first_xpath_text(trackpoint, "./tcx:Time", nsmap)
        if not time_text:
            continue

        timestamp = pd.to_datetime(time_text, utc=True, errors="coerce")
        if pd.isna(timestamp):
            continue

        hr_text = get_first_xpath_text(trackpoint, "./tcx:HeartRateBpm/tcx:Value", nsmap)

        rows.append(
            {
                "timestamp": timestamp,
                "hr_bpm": safe_int(hr_text),
                "power_w": None,
                "speed_kmh": None,
            }
        )

    df_data = pd.DataFrame(rows)
    if df_data.empty:
        return df_data

    df_data = df_data.dropna(subset=["timestamp"]).sort_values("timestamp")
    return df_data


def extract_dataframe_from_file(filename: str, file_bytes: bytes) -> pd.DataFrame:
    lowered_name = filename.lower()
    if lowered_name.endswith(".fit"):
        return parse_fit_bytes(file_bytes)
    if lowered_name.endswith(".tcx"):
        return parse_tcx_bytes(file_bytes)
    return pd.DataFrame()


def dataframe_to_apex_series(df_data: pd.DataFrame, value_column: str, series_name: str) -> Dict[str, Any]:
    if df_data.empty:
        return {"name": series_name, "data": []}

    df_metric = df_data.dropna(subset=["timestamp", value_column])
    if df_metric.empty:
        return {"name": series_name, "data": []}

    timestamp_ms = (df_metric["timestamp"].astype("int64") // 1_000_000).tolist()
    values = df_metric[value_column].tolist()

    points: List[List[float]] = []
    for ts_value, metric_value in zip(timestamp_ms, values):
        if metric_value is None:
            continue
        points.append([int(ts_value), float(metric_value)])

    return {"name": series_name, "data": points}


def init_series_response() -> Dict[str, List[Dict[str, Any]]]:
    return {metric: [] for metric in SUPPORTED_METRICS}


@app.post("/api/parse")
async def parse_activity_files(files: List[UploadFile] = File(...)) -> Dict[str, Any]:
    series_by_metric = init_series_response()

    for uploaded_file in files:
        filename = uploaded_file.filename or "file"
        file_bytes = await uploaded_file.read()

        df_data = extract_dataframe_from_file(filename, file_bytes)
        if df_data.empty:
            continue

        for metric in SUPPORTED_METRICS:
            apex_series = dataframe_to_apex_series(df_data, metric, filename)
            if apex_series["data"]:
                series_by_metric[metric].append(apex_series)

    return {"series": series_by_metric}
