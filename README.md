# Reachy Mini Conversation App (macOS Fork)

**Fully local conversational AI for Reachy Mini robot** - combining lightweight speech recognition, text-to-speech, and local LLM with choreographed motion libraries.

This is a fork of [dwain-barnes/reachy_mini_conversation_app_local](https://github.com/dwain-barnes/reachy_mini_conversation_app_local) with changes to run on **macOS (Apple Silicon)** over wireless (GStreamer/WebRTC).

![Reachy Mini Dance](docs/assets/reachy_mini_dance.gif)

## What's Changed in This Fork

### macOS Wireless Support
- **`install_mac_wireless.sh`** — One-step installer that handles all the dependency conflicts for macOS:
  - Works around `libusb_package>=1.0.26.3` not existing on PyPI (not needed for wireless)
  - Fixes GStreamer Python dylib path mismatch between Homebrew and python.org layouts
  - Handles overly strict version caps in `gst-signalling`, `reachy_mini_dances_library`, etc.
  - Pre-downloads ASR (Distil-Whisper) and TTS (Kokoro) models
- **`run`** — Convenience script to activate venv and launch in wireless mode

### Live Transcript Viewer
- **`transcript_server.py`** — Lightweight HTTP + SSE server that streams the conversation in real time
- Accessible at `http://localhost:7862` — shows a chat-style view of user and assistant messages
- Integrated into `console.py` so transcripts are pushed automatically during conversation

### Bug Fixes
- **`main.py`** — Fixed `robot.client.get_status()["simulation_enabled"]` → `.simulation_enabled` (attribute access, not dict)
- **`openai_realtime.py`** — Fixed `asyncio.get_event_loop()` crash by using `asyncio.get_running_loop()` with a fallback
- **`console.py`** — Added `FULL_LOCAL_MODE` support to skip OpenAI API key checks when running fully local

## Features

- 🎯 **100% Local Operation** - No cloud dependencies, runs entirely on-device
- 🎤 **Real-time Audio** - Low-latency speech-to-text (Distil-Whisper) and text-to-speech (Kokoro)
- 🤖 **Local LLM** - Powered by Ollama or LM Studio for on-device conversation
- 💃 **Motion System** - Layered motion with dances, emotions, face-tracking, and speech-reactive movement
- 🎨 **Custom Personalities** - Easy profile system for different robot behaviors
- 📝 **Live Transcript** - Real-time chat viewer at `http://localhost:7862`

## Prerequisites

> [!IMPORTANT]
> **Install Reachy Mini SDK first**: [github.com/pollen-robotics/reachy_mini](https://github.com/pollen-robotics/reachy_mini/)
>
> Works with:
> - **Real hardware** - Physical Reachy Mini robot
> - **Simulator** - Virtual Reachy Mini for testing

## Quick Start (macOS Wireless)

### 1. Install

```bash
git clone https://github.com/saket424/reachy_mini_conversation_app_local.git
cd reachy_mini_conversation_app_local

# Run the macOS installer (handles all dependency workarounds)
./install_mac_wireless.sh
```

### 2. Install Local LLM

**Ollama (Recommended):**
```bash
brew install ollama
ollama pull phi-3-mini-4k-instruct
```

**Or LM Studio:**
- Download from [lmstudio.ai](https://lmstudio.ai)
- Load a GGUF model (e.g., Phi-3-mini)
- Start local server on port 1234

### 3. Configure

```bash
cp .env.example .env
nano .env
```

### 4. Run

```bash
source .venv/bin/activate
reachy-mini-conversation-app --wireless-version
```

Or simply:
```bash
./run
```

The live transcript viewer will be available at `http://localhost:7862`.

## Quick Start (Linux / General)

### 1. Install

```bash
git clone https://github.com/saket424/reachy_mini_conversation_app_local.git
cd reachy_mini_conversation_app_local
pip install -e "."
```

### 2. Run

**Console mode (headless):**
```bash
reachy-mini-conversation-app
```

**Web UI mode (required for simulator):**
```bash
reachy-mini-conversation-app --gradio
```

Access at `http://localhost:7860`

## Configuration

The app auto-configures for your hardware. Key settings in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `ollama` | LLM backend (`ollama` or `lmstudio`) |
| `OLLAMA_MODEL` | `phi-3-mini-4k-instruct` | Ollama model name |
| `DISTIL_WHISPER_MODEL` | `distil-small.en` | Speech recognition model |
| `KOKORO_VOICE` | `af_sarah` | TTS voice (af_sarah, am_michael, etc.) |
| `FULL_LOCAL_MODE` | `false` | Skip OpenAI API key requirement |

## CLI Options

| Option | Description |
|--------|-------------|
| `--gradio` | Launch web UI (required for simulator) |
| `--head-tracker {yolo,mediapipe}` | Enable face tracking |
| `--local-vision` | Use local vision model (requires `local_vision` extra) |
| `--no-camera` | Disable camera (audio-only mode) |
| `--wireless-version` | Use GStreamer for wireless robots |
| `--debug` | Enable verbose logging |

## Optional Extras

```bash
# Vision features
pip install -e ".[local_vision]"      # Local vision model (SmolVLM2)
pip install -e ".[yolo_vision]"       # YOLO face tracking
pip install -e ".[mediapipe_vision]"  # MediaPipe tracking
pip install -e ".[all_vision]"        # All vision features

# Hardware support
pip install -e ".[reachy_mini_wireless]"  # Wireless Reachy Mini
pip install -e ".[jetson]"                 # Jetson optimization (CUDA)

# Development
pip install -e ".[dev]"  # Testing & linting tools
```

## Available Tools

The LLM has access to these robot actions:

| Tool | Action |
|------|--------|
| `move_head` | Move head (left/right/up/down/front) |
| `camera` | Capture and analyze camera image |
| `head_tracking` | Enable/disable face tracking |
| `dance` | Play choreographed dance |
| `stop_dance` | Stop current dance |
| `play_emotion` | Display emotion animation |
| `stop_emotion` | Stop emotion animation |
| `do_nothing` | Remain idle |

## Custom Personalities

Create custom robot personalities with unique behaviors:

1. Set profile name: `REACHY_MINI_CUSTOM_PROFILE=my_profile` in `.env`
2. Create folder: `src/reachy_mini_conversation_app/profiles/my_profile/`
3. Add files:
   - `instructions.txt` - Personality prompt
   - `tools.txt` - Available tools (one per line)
   - `custom_tool.py` - Optional custom tools

See `profiles/example/` for reference.

## Architecture

```
User Speech → VAD → Distil-Whisper STT → Local LLM → Kokoro TTS → Audio Output
                                              ↓
                                         Tool Dispatch
                                              ↓
                                    Robot Actions (Motion/Vision)
```

All processing runs locally using:
- **VAD**: Built-in energy-based detection
- **STT**: Distil-Whisper (lightweight, 2-6x faster)
- **LLM**: Ollama/LM Studio (Phi-3-mini recommended)
- **TTS**: Kokoro-82M via FastRTC (production quality)
- **Framework**: FastRTC for low-latency audio streaming

## Troubleshooting

**TimeoutError connecting to robot:**
```bash
# Start the Reachy Mini daemon first
# See: https://github.com/pollen-robotics/reachy_mini/
```

**No audio output:**
- Check TTS voice is valid: `af_sarah`, `am_michael`, `bf_emma`, `bm_lewis`
- Verify Ollama/LM Studio is running: `curl http://localhost:11434` or `:1234`

**GStreamer plugin errors on macOS:**
- Re-run `install_mac_wireless.sh` — it creates the needed Python framework symlink
- Check that `gstreamer-bundle` is installed: `pip list | grep gstreamer`

## License

Apache 2.0

---

**Built for edge deployment** - Optimized for any hardware with 8GB+ RAM.

Thanks to [dwain-barnes](https://github.com/dwain-barnes) for the original app and [muellerzr](https://github.com/muellerzr) for his fork.
