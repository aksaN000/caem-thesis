#!/usr/bin/env bash
# scripts/expose_demo_remote.sh
# =============================
# Exposes localhost:8000 (the CAEM demo server) to a public URL via
# Cloudflare's free `cloudflared` tunnel. Use for hybrid/remote defense
# panels or when sharing the demo with a remote friend.
#
# Prerequisite: cloudflared installed
#   Ubuntu/Debian: sudo apt-get install cloudflared
#   macOS:         brew install cloudflared
#   Manual:        https://github.com/cloudflare/cloudflared/releases
#
# No Cloudflare account needed for the ephemeral tunnel mode used here.
# The URL changes every time the tunnel restarts; share the URL printed
# in the terminal output (look for "https://...trycloudflare.com").
#
# Usage:
#   bash scripts/expose_demo_remote.sh
#
# To stop: Ctrl+C in this terminal. The tunnel dies with the process.

set -eu

PORT="${1:-8000}"
LOCAL_URL="http://localhost:${PORT}"

# Sanity-check the demo server is actually listening before exposing
if ! curl -sSf -o /dev/null --max-time 3 "${LOCAL_URL}/health"; then
    echo "ERROR: nothing listening on ${LOCAL_URL}/health"
    echo "Start the demo server first:"
    echo "  python scripts/caem_demo_server.py --memory <path> --passage_index <path> --port ${PORT}"
    echo "Then re-run this script."
    exit 1
fi

if ! command -v cloudflared >/dev/null 2>&1; then
    echo "ERROR: cloudflared not installed"
    echo "Ubuntu/Debian: sudo apt-get install cloudflared"
    echo "macOS:         brew install cloudflared"
    echo "Or download from: https://github.com/cloudflare/cloudflared/releases"
    exit 1
fi

echo "[CAEM-tunnel] starting cloudflared ephemeral tunnel for ${LOCAL_URL}"
echo "[CAEM-tunnel] the public URL will be printed below — share it with the panel"
echo "[CAEM-tunnel] press Ctrl+C in this terminal to stop the tunnel"
echo ""

exec cloudflared tunnel --url "${LOCAL_URL}"
