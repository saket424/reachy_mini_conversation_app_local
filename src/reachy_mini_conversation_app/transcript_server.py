"""Lightweight live transcript viewer served over HTTP + SSE."""

import asyncio
import json
import logging
import threading
import time
from collections import deque
from typing import Any, Dict, List

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from starlette.responses import StreamingResponse

logger = logging.getLogger(__name__)

_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Reachy Mini – Live Transcript</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
      integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=" crossorigin="">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
        integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=" crossorigin=""></script>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background: #0f0f0f; color: #e0e0e0; padding: 1rem;
  }
  h1 { font-size: 1.1rem; color: #888; margin-bottom: 1rem; }
  #status { font-size: 0.8rem; color: #555; margin-bottom: 0.5rem; }
  #transcript {
    display: flex; flex-direction: column; gap: 0.6rem;
    max-width: 720px; margin: 0 auto;
  }
  .msg {
    padding: 0.6rem 0.9rem; border-radius: 0.75rem;
    max-width: 85%; line-height: 1.45; font-size: 0.95rem;
    animation: fadein 0.2s ease-in;
  }
  .msg.user {
    align-self: flex-end; background: #1a3a5c; color: #d0e8ff;
    border-bottom-right-radius: 0.2rem;
  }
  .msg.assistant {
    align-self: flex-start; background: #2a2a2a; color: #e0e0e0;
    border-bottom-left-radius: 0.2rem;
  }
  .msg .role { font-size: 0.7rem; color: #888; margin-bottom: 0.15rem; }
  .msg .ts   { font-size: 0.65rem; color: #555; margin-top: 0.2rem; text-align: right; }
  .card {
    align-self: stretch; max-width: 100%; padding: 0.6rem 0.9rem;
    background: #1c2630; color: #e0e0e0; border-radius: 0.75rem;
    border: 1px solid #2a3a4a; animation: fadein 0.2s ease-in;
  }
  .card .title { font-size: 0.85rem; color: #9ec5ff; margin-bottom: 0.4rem; }
  .card .map   { width: 100%; height: 280px; border-radius: 0.5rem; margin: 0.4rem 0; }
  .card .kv    { font-size: 0.85rem; color: #cfd6dd; }
  .card .kv div { display: flex; justify-content: space-between; padding: 0.15rem 0; border-bottom: 1px dashed #2a3a4a; }
  .card .kv div:last-child { border-bottom: none; }
  .card .kv .k { color: #8fa3b8; }
  .card .kv .v { color: #e8eef5; font-variant-numeric: tabular-nums; }
  .card .ts    { font-size: 0.65rem; color: #555; margin-top: 0.4rem; text-align: right; }
  @keyframes fadein { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: none; } }
</style>
</head>
<body>
<h1>Reachy Mini – Live Transcript</h1>
<div id="status">connecting…</div>
<div id="transcript"></div>
<script>
const tx = document.getElementById("transcript");
const st = document.getElementById("status");
let mapSeq = 0;

function addMsg(d) {
  if (d.kind === "map") return addMap(d);
  const el = document.createElement("div");
  el.className = "msg " + (d.role || "assistant");
  const rl = document.createElement("div");
  rl.className = "role";
  rl.textContent = d.role === "user" ? "You" : "Reachy";
  const ct = document.createElement("div");
  ct.textContent = d.content;
  el.appendChild(rl);
  el.appendChild(ct);
  if (d.time) {
    const ts = document.createElement("div");
    ts.className = "ts";
    ts.textContent = d.time;
    el.appendChild(ts);
  }
  tx.appendChild(el);
  window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
}

function addMap(d) {
  const el = document.createElement("div");
  el.className = "card";
  const title = document.createElement("div");
  title.className = "title";
  title.textContent = d.title || "Map";
  el.appendChild(title);

  const mapId = "map-" + (++mapSeq);
  const mapDiv = document.createElement("div");
  mapDiv.className = "map";
  mapDiv.id = mapId;
  el.appendChild(mapDiv);

  if (Array.isArray(d.values) && d.values.length) {
    const kv = document.createElement("div");
    kv.className = "kv";
    d.values.forEach(([k, v]) => {
      const row = document.createElement("div");
      const ks = document.createElement("span"); ks.className = "k"; ks.textContent = k;
      const vs = document.createElement("span"); vs.className = "v"; vs.textContent = v;
      row.appendChild(ks); row.appendChild(vs);
      kv.appendChild(row);
    });
    el.appendChild(kv);
  }

  if (d.time) {
    const ts = document.createElement("div");
    ts.className = "ts";
    ts.textContent = d.time;
    el.appendChild(ts);
  }

  tx.appendChild(el);

  // Init Leaflet after the div is in the DOM.
  const tileUrl = "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png";
  const tileOpts = { maxZoom: 19, attribution: '&copy; OpenStreetMap' };

  if (Array.isArray(d.markers) && d.markers.length) {
    const m = L.map(mapId, { scrollWheelZoom: false });
    L.tileLayer(tileUrl, tileOpts).addTo(m);
    const layers = [];
    d.markers.forEach((mk) => {
      const la = Number(mk.lat), lo = Number(mk.lon);
      if (!Number.isFinite(la) || !Number.isFinite(lo)) return;
      const marker = L.marker([la, lo]).addTo(m);
      if (mk.popup) marker.bindPopup(mk.popup);
      layers.push(marker);
    });
    if (layers.length === 1) {
      m.setView(layers[0].getLatLng(), d.zoom || 10);
      if (d.markers[0].popup) layers[0].openPopup();
    } else if (layers.length > 1) {
      m.fitBounds(L.featureGroup(layers).getBounds().pad(0.15));
    } else {
      m.setView([39.5, -98.35], 4);  // fallback: continental US
    }
  } else {
    const lat = Number(d.lat), lon = Number(d.lon);
    if (Number.isFinite(lat) && Number.isFinite(lon)) {
      const m = L.map(mapId, { scrollWheelZoom: false }).setView([lat, lon], d.zoom || 13);
      L.tileLayer(tileUrl, tileOpts).addTo(m);
      const marker = L.marker([lat, lon]).addTo(m);
      if (d.popup) marker.bindPopup(d.popup).openPopup();
    }
  }

  window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
}

function connect() {
  const es = new EventSource("/events");
  es.onopen = () => { st.textContent = "connected"; };
  es.onmessage = (e) => {
    try { addMsg(JSON.parse(e.data)); } catch(_) {}
  };
  es.onerror = () => {
    st.textContent = "reconnecting…";
    es.close();
    setTimeout(connect, 2000);
  };
}
fetch("/history").then(r => r.json()).then(msgs => {
  msgs.forEach(addMsg);
  connect();
});
</script>
</body>
</html>
"""

MAX_HISTORY = 200


class TranscriptServer:
    """Runs a tiny HTTP server that streams transcript messages via SSE."""

    def __init__(self, port: int = 7862):
        self._port = port
        self._history: deque[Dict[str, Any]] = deque(maxlen=MAX_HISTORY)
        self._subscribers: List[asyncio.Queue[Dict[str, Any]]] = []
        self._lock = threading.Lock()
        self._app = self._build_app()
        self._thread: threading.Thread | None = None

    def push(self, role: str, content: str) -> None:
        entry = {
            "kind": "text",
            "role": role,
            "content": content,
            "time": time.strftime("%H:%M:%S"),
        }
        self._broadcast(entry)

    def push_card(self, payload: Dict[str, Any]) -> None:
        """Push a structured card (e.g. Leaflet map) to the viewer."""
        entry = {
            "kind": payload.get("kind", "map"),
            "time": time.strftime("%H:%M:%S"),
            **{k: v for k, v in payload.items() if k != "kind"},
        }
        self._broadcast(entry)

    def _broadcast(self, entry: Dict[str, Any]) -> None:
        with self._lock:
            self._history.append(entry)
            for q in self._subscribers:
                try:
                    q.put_nowait(entry)
                except asyncio.QueueFull:
                    pass

    def _build_app(self) -> FastAPI:
        app = FastAPI()

        @app.get("/", response_class=HTMLResponse)
        async def _index() -> str:
            return _HTML

        @app.get("/history")
        async def _history() -> List[Dict[str, Any]]:
            with self._lock:
                return list(self._history)

        @app.get("/events")
        async def _events(request: Request) -> StreamingResponse:
            q: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(maxsize=64)
            with self._lock:
                self._subscribers.append(q)

            async def stream():
                try:
                    while True:
                        if await request.is_disconnected():
                            break
                        try:
                            entry = await asyncio.wait_for(q.get(), timeout=15)
                            yield f"data: {json.dumps(entry)}\n\n"
                        except asyncio.TimeoutError:
                            yield ": keepalive\n\n"
                finally:
                    with self._lock:
                        try:
                            self._subscribers.remove(q)
                        except ValueError:
                            pass

            return StreamingResponse(stream(), media_type="text/event-stream")

        return app

    def start(self) -> None:
        def _run() -> None:
            uvicorn.run(self._app, host="0.0.0.0", port=self._port, log_level="warning")

        self._thread = threading.Thread(target=_run, daemon=True, name="transcript-server")
        self._thread.start()
        logger.info("Transcript viewer at http://localhost:%d", self._port)
