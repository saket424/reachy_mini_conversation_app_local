#!/usr/bin/env bash
#
# install_mac_wireless.sh
#
# Installs reachy_mini_conversation_app on macOS for wireless (GStreamer/WebRTC)
# usage with a Reachy Mini robot.
#
# Problem: reachy_mini requires libusb_package>=1.0.26.3, but only 1.0.26.1
# exists on PyPI. This is a hard block for pip's resolver. Since wireless mode
# doesn't need USB at all, we install reachy_mini with --no-deps and then
# install its actual runtime dependencies separately.
#
# Additional issues handled:
#   - reachy_mini_dances_library and reachy_mini_toolbox pull scipy through
#     a chain that tries to build from source on Python 3.14; installing them
#     with --no-deps avoids this since scipy is already installed as a wheel.
#   - gst-signalling has overly strict upper bounds on numpy and PyGObject;
#     --no-deps is used since the actual APIs are compatible.
#   - gradio 5.50.1.dev1 (pinned in pyproject.toml) may not be available;
#     fastrtc pulls in 5.50.0 which is functionally equivalent.
#   - gstreamer-bundle's libgstpython.dylib is compiled against the
#     python.org framework layout (/Library/Frameworks/Python.framework/...)
#     but Homebrew installs Python under /opt/homebrew/. A symlink is needed
#     so GStreamer can find the Python dylib at runtime.
#
# Tested with: Python 3.14.4, macOS (Apple Silicon), 2026-04-23

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# 1. Create virtual environment
# ---------------------------------------------------------------------------
if [ ! -d ".venv" ]; then
    echo "==> Creating virtual environment..."
    python3 -m venv .venv
else
    echo "==> .venv already exists, reusing it."
fi

source .venv/bin/activate
echo "==> Using Python: $(python3 --version) at $(which python3)"

# ---------------------------------------------------------------------------
# 2. Upgrade pip
# ---------------------------------------------------------------------------
echo "==> Upgrading pip..."
pip install --upgrade pip

# ---------------------------------------------------------------------------
# 3. Install reachy_mini with --no-deps
#    Reason: reachy_mini requires libusb_package>=1.0.26.3 which does not
#    exist on PyPI (latest is 1.0.26.1). This constraint is baked into
#    reachy_mini's package metadata and cannot be overridden by pip.
#    Since we are using wireless (not USB), libusb is not needed at runtime.
# ---------------------------------------------------------------------------
echo "==> Installing reachy_mini (--no-deps to skip libusb_package>=1.0.26.3)..."
pip install reachy_mini --no-deps

# ---------------------------------------------------------------------------
# 4. Install reachy_mini's actual dependencies (minus libusb_package)
#    These are the deps listed in reachy_mini's metadata, excluding:
#      - libusb_package (unavailable, not needed for wireless)
#      - gstreamer-bundle (installed separately below, version may differ)
# ---------------------------------------------------------------------------
echo "==> Installing reachy_mini runtime dependencies..."
pip install \
    aiohttp \
    asgiref \
    fastapi \
    huggingface-hub \
    jinja2 \
    log-throttling \
    numpy \
    pip \
    psutil \
    pyserial \
    python-multipart \
    pyusb \
    pyyaml \
    questionary \
    reachy-mini-rust-kinematics \
    reachy_mini_motor_controller \
    requests \
    rich \
    rustypot \
    scipy \
    starlette \
    toml \
    tornado \
    uvicorn \
    websockets \
    zeroconf

# ---------------------------------------------------------------------------
# 5. Install GStreamer bundle (for wireless WebRTC support)
#    reachy_mini pins ==1.28.1 but only 1.28.2 may be available; this is fine.
# ---------------------------------------------------------------------------
echo "==> Installing gstreamer-bundle..."
pip install gstreamer-bundle

# ---------------------------------------------------------------------------
# 6. Install gst-signalling (for wireless WebRTC signalling)
#    Uses --no-deps because it has overly strict upper bounds:
#      - numpy<=2.2.5 (we have 2.4.x, works fine)
#      - PyGObject<=3.49.0 (gstreamer-bundle installs 3.50.x, works fine)
# ---------------------------------------------------------------------------
echo "==> Installing gst-signalling (--no-deps to avoid numpy/PyGObject cap)..."
pip install "gst-signalling>=1.1.2" --no-deps

# ---------------------------------------------------------------------------
# 7. Install reachy_mini_dances_library and reachy_mini_toolbox
#    Uses --no-deps because their dependency chains try to rebuild scipy
#    from source on Python 3.14 (no Fortran compiler). scipy is already
#    installed as a binary wheel from step 4.
# ---------------------------------------------------------------------------
echo "==> Installing reachy_mini_dances_library and reachy_mini_toolbox (--no-deps)..."
pip install reachy_mini_dances_library --no-deps
pip install reachy_mini_toolbox --no-deps

# ---------------------------------------------------------------------------
# 8. Install the remaining project dependencies
#    These install cleanly without workarounds.
# ---------------------------------------------------------------------------
echo "==> Installing media and ML dependencies..."
pip install \
    aiortc \
    "fastrtc>=0.0.34" \
    "opencv-python>=4.12.0.88" \
    python-dotenv \
    "eclipse-zenoh~=1.7.0" \
    distil-whisper-fastrtc \
    torch \
    transformers \
    accelerate \
    openai \
    kokoro-onnx \
    torchaudio

# ---------------------------------------------------------------------------
# 9. Fix GStreamer Python dylib path (Homebrew Python only)
#    gstreamer-bundle ships libgstpython.dylib compiled to look for Python at
#    /Library/Frameworks/Python.framework/Versions/3.14/Python (the python.org
#    installer layout). Homebrew installs the framework under /opt/homebrew/,
#    so GStreamer fails to load the plugin at runtime with:
#      "Library not loaded: /Library/Frameworks/Python.framework/Versions/3.14/Python"
#    A symlink bridges the two locations. Requires sudo.
# ---------------------------------------------------------------------------
PYTHON_MINOR=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
FRAMEWORK_TARGET="/Library/Frameworks/Python.framework/Versions/${PYTHON_MINOR}/Python"
HOMEBREW_FRAMEWORK="/opt/homebrew/Frameworks/Python.framework/Versions/${PYTHON_MINOR}/Python"

if [ ! -f "$FRAMEWORK_TARGET" ] && [ -f "$HOMEBREW_FRAMEWORK" ]; then
    echo "==> Creating symlink for GStreamer Python dylib (requires sudo)..."
    sudo mkdir -p "$(dirname "$FRAMEWORK_TARGET")"
    sudo ln -s "$HOMEBREW_FRAMEWORK" "$FRAMEWORK_TARGET"
    echo "  Linked $FRAMEWORK_TARGET -> $HOMEBREW_FRAMEWORK"
elif [ -f "$FRAMEWORK_TARGET" ]; then
    echo "==> Python framework dylib already exists at $FRAMEWORK_TARGET, skipping."
else
    echo "==> WARNING: Homebrew Python framework not found at $HOMEBREW_FRAMEWORK"
    echo "   GStreamer's libgstpython.dylib may fail to load."
    echo "   If you installed Python via python.org this is fine."
fi

# ---------------------------------------------------------------------------
# 10. Install the project itself in editable mode (--no-deps since everything
#     is already installed; avoids re-triggering the resolver)
# ---------------------------------------------------------------------------
echo "==> Installing reachy_mini_conversation_app (editable)..."
pip install -e "." --no-deps

# ---------------------------------------------------------------------------
# 11. Pre-download ASR and TTS models
#     These are lazy-loaded on first use. Downloading them now avoids a long
#     delay on the first spoken utterance. Models are cached under
#     ~/.cache/huggingface/hub/
# ---------------------------------------------------------------------------
echo "==> Pre-downloading Distil-Whisper (ASR/STT) model..."
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('distil-whisper/distil-small.en')"

echo "==> Pre-downloading Kokoro (TTS) model..."
python3 -c "from fastrtc import get_tts_model; get_tts_model(model='kokoro', voice='af_sarah')"

# ---------------------------------------------------------------------------
# 12. Verify critical imports
# ---------------------------------------------------------------------------
echo "==> Verifying imports..."
python3 -c "
import reachy_mini_conversation_app
print('  [OK] reachy_mini_conversation_app')

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
print('  [OK] GStreamer (gi.repository.Gst)')

from gst_signalling import GstSignallingProducer
print('  [OK] gst-signalling')

import torch
print('  [OK] torch')

import transformers
print('  [OK] transformers')
"

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo "==> Installation complete."
echo ""
echo "Known harmless version mismatches (shown by 'pip check'):"
echo "  - libusb_package: 1.0.26.1 installed, >=1.0.26.3 required (USB only, not needed)"
echo "  - gstreamer-bundle: minor version difference from reachy_mini pin"
echo "  - huggingface-hub: newer than reachy_mini's pin (backward compatible)"
echo "  - gradio: 5.50.0 vs 5.50.1.dev1 (dev pin, functionally equivalent)"
echo "  - numpy/PyGObject: newer than gst-signalling's caps (APIs compatible)"
echo ""
echo "To run: source .venv/bin/activate && reachy-mini-conversation-app"
