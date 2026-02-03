import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple


SCENE_KEYWORDS = {
    "烟火": "烟火",
    "火灾": "火灾",
    "盗采": "盗采",
    "违法建设": "违法建设",
    "违建": "违法建设",
    "施工": "施工",
    "烟雾": "烟雾",
}


@dataclass
class QueryPlan:
    intent: str
    sql: str
    params: List
    filters: Dict


def _parse_top_k(text: str, default: int = 20) -> int:
    match = re.search(r"(前|top|TOP)\s*(\d+)", text)
    if match:
        return int(match.group(2))
    match = re.search(r"(\d+)\s*条", text)
    if match:
        return int(match.group(1))
    return default


def _parse_time_range(text: str) -> Tuple[Optional[str], Optional[str]]:
    date_matches = re.findall(r"\d{4}-\d{2}-\d{2}", text)
    if len(date_matches) >= 2:
        return date_matches[0], date_matches[1]

    match = re.search(r"近(\d+)(天|小时)", text)
    if match:
        value = int(match.group(1))
        unit = match.group(2)
        end = datetime.utcnow()
        start = end - (timedelta(days=value) if unit == "天" else timedelta(hours=value))
        return start.isoformat(timespec="seconds"), end.isoformat(timespec="seconds")
    return None, None


def _parse_location(text: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    lat = None
    lon = None
    radius = None
    lat_match = re.search(r"lat\s*=\s*([0-9.]+)", text, re.IGNORECASE)
    lon_match = re.search(r"lon\s*=\s*([0-9.]+)", text, re.IGNORECASE)
    if lat_match:
        lat = float(lat_match.group(1))
    if lon_match:
        lon = float(lon_match.group(1))
    radius_match = re.search(r"(\d+)\s*公里", text)
    if radius_match:
        radius = float(radius_match.group(1))
    return lat, lon, radius


def _parse_intent(text: str) -> str:
    if any(k in text for k in ["多少", "统计", "数量", "总数"]):
        return "count"
    return "list"


def parse_question(text: str) -> QueryPlan:
    intent = _parse_intent(text)
    event_type = None
    for key, value in SCENE_KEYWORDS.items():
        if key in text:
            event_type = value
            break

    start_time, end_time = _parse_time_range(text)
    top_k = _parse_top_k(text)
    lat, lon, radius_km = _parse_location(text)

    where = []
    params: List = []
    if event_type:
        where.append("e.event_type = ?")
        params.append(event_type)
    if start_time:
        where.append("e.alarm_time >= ?")
        params.append(start_time)
    if end_time:
        where.append("e.alarm_time <= ?")
        params.append(end_time)

    where_sql = " WHERE " + " AND ".join(where) if where else ""

    if intent == "count":
        sql = (
            "SELECT COUNT(*) AS cnt FROM events e "
            "LEFT JOIN assets a ON e.asset_id = a.asset_id"
            + where_sql
        )
    else:
        sql = (
            "SELECT e.event_id, e.event_type, e.alarm_time, a.file_path, a.lat, a.lon "
            "FROM events e LEFT JOIN assets a ON e.asset_id = a.asset_id"
            + where_sql
            + " ORDER BY e.alarm_time DESC LIMIT ?"
        )
        params.append(top_k)

    filters = {
        "event_type": event_type,
        "start_time": start_time,
        "end_time": end_time,
        "lat": lat,
        "lon": lon,
        "radius_km": radius_km,
        "top_k": top_k,
    }

    return QueryPlan(intent=intent, sql=sql, params=params, filters=filters)
