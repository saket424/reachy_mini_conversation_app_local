"""Wildfire activity tool: query NIFC WFIGS active incidents and render on a map."""

import asyncio
import json
import logging
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


WFIGS_URL = (
    "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/"
    "WFIGS_Incident_Locations_Current/FeatureServer/0/query"
)

OUT_FIELDS = ",".join([
    "IncidentName",
    "IncidentSize",
    "PercentContained",
    "POOState",
    "POOCity",
    "IncidentTypeCategory",
    "FireDiscoveryDateTime",
])


def _http_get(url: str, timeout: int = 20) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "reachy-mini-wildfire-tool/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _query_wfigs(state: Optional[str], min_acres: float, limit: int) -> List[Dict[str, Any]]:
    """Return list of GeoJSON features for active wildfires matching the filter."""
    where_clauses = ["IncidentTypeCategory='WF'", f"IncidentSize>{min_acres}"]
    if state:
        where_clauses.append(f"POOState='US-{state.upper()}'")
    where = " AND ".join(where_clauses)

    params = {
        "where": where,
        "outFields": OUT_FIELDS,
        "f": "geojson",
        "returnGeometry": "true",
        "orderByFields": "IncidentSize DESC",
        "resultRecordCount": str(limit),
    }
    url = f"{WFIGS_URL}?{urllib.parse.urlencode(params)}"

    try:
        body = _http_get(url)
        data = json.loads(body)
    except Exception as e:
        logger.warning("WFIGS query failed: %s", e)
        return []

    return data.get("features", []) or []


class WildfireActivity(Tool):
    """Look up current US wildfire activity from NIFC and display on a map."""

    name = "wildfire_activity"
    description = (
        "Look up active US wildfire incidents from NIFC's public WFIGS feed and display them "
        "as markers on an interactive map (visible to the user in the live transcript viewer). "
        "Optionally filter by 2-letter US state code (e.g. 'CA', 'OR') and minimum acreage. "
        "Returns a brief spoken summary you should voice succinctly first; richer per-incident "
        "details are available in the response if the user follows up."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "state": {
                "type": "string",
                "description": "Optional two-letter US state code to filter by (e.g. 'CA'). Omit for nationwide.",
            },
            "min_acres": {
                "type": "number",
                "description": "Minimum incident size in acres. Default 100. Use a larger value (e.g. 1000) for big-fire-only summaries.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of incidents to return and plot. Default 15.",
            },
        },
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        state: Optional[str] = (kwargs.get("state") or "").strip() or None
        min_acres: float = float(kwargs.get("min_acres", 100) or 100)
        limit: int = int(kwargs.get("limit", 15) or 15)
        limit = max(1, min(limit, 50))

        logger.info("Tool call: wildfire_activity state=%r min_acres=%s limit=%s",
                    state, min_acres, limit)

        features = await asyncio.to_thread(_query_wfigs, state, min_acres, limit)

        scope = state.upper() if state else "the US"
        if not features:
            summary = (f"No active wildfires over {int(min_acres)} acres reported in {scope} "
                       f"in NIFC's current feed.")
            return {"status": "ok", "scope": scope, "count": 0, "incidents": [], "summary": summary}

        incidents: List[Dict[str, Any]] = []
        markers: List[Dict[str, Any]] = []
        total_acres = 0.0

        for feat in features:
            geom = feat.get("geometry") or {}
            coords = geom.get("coordinates") or []
            if len(coords) < 2:
                continue
            lon, lat = coords[0], coords[1]
            props = feat.get("properties") or {}
            name = props.get("IncidentName") or "Unnamed incident"
            size = props.get("IncidentSize") or 0
            try:
                size = float(size)
            except (TypeError, ValueError):
                size = 0.0
            total_acres += size
            contained = props.get("PercentContained")
            poo_state = (props.get("POOState") or "").replace("US-", "")
            poo_city = props.get("POOCity") or ""

            incident = {
                "name": name,
                "acres": size,
                "percent_contained": contained,
                "state": poo_state,
                "city": poo_city,
                "lat": lat,
                "lon": lon,
            }
            incidents.append(incident)

            popup_lines = [f"<b>{name}</b>"]
            if poo_city or poo_state:
                where = ", ".join(p for p in (poo_city, poo_state) if p)
                popup_lines.append(where)
            popup_lines.append(f"{int(size):,} acres")
            if contained is not None:
                popup_lines.append(f"{int(contained)}% contained")
            markers.append({"lat": lat, "lon": lon, "popup": "<br>".join(popup_lines)})

        # Push the map card.
        if deps.transcript_server is not None and markers:
            title = f"Active wildfires – {scope} ({len(incidents)} shown)"
            try:
                deps.transcript_server.push_card({
                    "kind": "map",
                    "title": title,
                    "markers": markers,
                    "values": [
                        ["Incidents shown", str(len(incidents))],
                        ["Total acres", f"{int(total_acres):,}"],
                        ["Largest", f"{incidents[0]['name']} ({int(incidents[0]['acres']):,} ac)"],
                    ],
                })
            except Exception as e:
                logger.warning("Failed to push wildfire map card: %s", e)

        # Succinct first-pass summary for voice playback.
        biggest = incidents[0]
        contained_str = (f", {int(biggest['percent_contained'])}% contained"
                         if biggest['percent_contained'] is not None else "")
        summary = (
            f"{len(incidents)} active wildfires over {int(min_acres)} acres in {scope}, "
            f"about {int(total_acres):,} acres total. Largest is the {biggest['name']} fire "
            f"at {int(biggest['acres']):,} acres{contained_str}."
        )

        return {
            "status": "ok",
            "scope": scope,
            "count": len(incidents),
            "total_acres": int(total_acres),
            "incidents": incidents,
            "summary": summary,
        }
