"""Home Assistant tool for querying and controlling smart home devices."""

import asyncio
import json
import logging
import os
import urllib.request
import urllib.error
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

HA_URL = os.getenv("HOME_ASSISTANT_URL", "").rstrip("/")
HA_TOKEN = os.getenv("HOME_ASSISTANT_TOKEN", "")


def _ha_request(path: str, method: str = "GET", body: dict | None = None) -> dict:
    if not HA_URL or not HA_TOKEN:
        return {"error": "Home Assistant not configured. Set HOME_ASSISTANT_URL and HOME_ASSISTANT_TOKEN."}

    url = f"{HA_URL}/api/{path}"
    headers = {
        "Authorization": f"Bearer {HA_TOKEN}",
        "Content-Type": "application/json",
    }
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.reason}"}
    except urllib.error.URLError as e:
        return {"error": f"Connection failed: {e.reason}"}
    except Exception as e:
        return {"error": str(e)}


class HomeAssistant(Tool):
    """Query or control Home Assistant devices."""

    name = "home_assistant"
    description = (
        "Interact with Home Assistant smart home. Use action 'get_states' to list all device states, "
        "'get_state' with an entity_id to check a specific device, or 'call_service' to control devices "
        "(e.g. turn on/off lights, switches). Available entities include sensors for temperature, humidity, "
        "energy, power, and robot telemetry."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["get_states", "get_state", "call_service"],
                "description": (
                    "Action to perform: 'get_states' lists all entities, "
                    "'get_state' reads one entity, 'call_service' controls a device."
                ),
            },
            "entity_id": {
                "type": "string",
                "description": "Entity ID (e.g. 'sensor.temperature', 'switch.shelly_plug'). Required for get_state and call_service.",
            },
            "domain": {
                "type": "string",
                "description": "Service domain (e.g. 'switch', 'light', 'climate'). Required for call_service.",
            },
            "service": {
                "type": "string",
                "description": "Service to call (e.g. 'turn_on', 'turn_off', 'toggle'). Required for call_service.",
            },
            "service_data": {
                "type": "object",
                "description": "Additional data for the service call (e.g. {'brightness': 255}).",
            },
        },
        "required": ["action"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        action = kwargs.get("action", "")
        logger.info("Tool call: home_assistant action=%s", action)

        if action == "get_states":
            return await self._get_states()
        elif action == "get_state":
            entity_id = kwargs.get("entity_id", "")
            if not entity_id:
                return {"error": "entity_id is required for get_state"}
            return await self._get_state(entity_id)
        elif action == "call_service":
            return await self._call_service(
                domain=kwargs.get("domain", ""),
                service=kwargs.get("service", ""),
                entity_id=kwargs.get("entity_id"),
                service_data=kwargs.get("service_data"),
            )
        else:
            return {"error": f"Unknown action: {action}. Use get_states, get_state, or call_service."}

    async def _get_states(self) -> Dict[str, Any]:
        result = await asyncio.to_thread(_ha_request, "states")
        if isinstance(result, dict) and "error" in result:
            return result

        if not isinstance(result, list):
            return {"error": "Unexpected response from Home Assistant"}

        summary = []
        for entity in result:
            eid = entity.get("entity_id", "")
            state = entity.get("state", "")
            attrs = entity.get("attributes", {})
            friendly = attrs.get("friendly_name", eid)
            unit = attrs.get("unit_of_measurement", "")
            summary.append(f"{friendly} ({eid}): {state}{(' ' + unit) if unit else ''}")

        return {"status": "ok", "entity_count": len(result), "entities": "\n".join(summary)}

    async def _get_state(self, entity_id: str) -> Dict[str, Any]:
        result = await asyncio.to_thread(_ha_request, f"states/{entity_id}")
        if isinstance(result, dict) and "error" in result:
            return result

        state = result.get("state", "unknown")
        attrs = result.get("attributes", {})
        friendly = attrs.get("friendly_name", entity_id)
        unit = attrs.get("unit_of_measurement", "")
        last_changed = result.get("last_changed", "")

        return {
            "status": "ok",
            "entity_id": entity_id,
            "friendly_name": friendly,
            "state": state,
            "unit": unit,
            "last_changed": last_changed,
            "attributes": {k: v for k, v in attrs.items() if k not in ("friendly_name", "unit_of_measurement")},
        }

    async def _call_service(
        self, domain: str, service: str, entity_id: str | None, service_data: dict | None
    ) -> Dict[str, Any]:
        if not domain or not service:
            return {"error": "domain and service are required for call_service"}

        body: dict = {}
        if entity_id:
            body["entity_id"] = entity_id
        if service_data:
            body.update(service_data)

        result = await asyncio.to_thread(_ha_request, f"services/{domain}/{service}", method="POST", body=body)
        if isinstance(result, dict) and "error" in result:
            return result

        return {"status": "ok", "service": f"{domain}.{service}", "entity_id": entity_id or "all"}
