#!/usr/bin/env bash
# SortOps on the robot PC: the cell API + web UI on http://127.0.0.1:8200, run with Physical AI Studio's backend
# venv (physicalai, SO-101 plugin, OpenVINO already there). systemd --user unit "sortops".
# Usage: run.sh install | start | stop | restart | status | logs | open | foreground | build
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDIO="${STUDIO_DIR:-$HOME/hackathon/physical-ai-studio}"
PY="$STUDIO/application/backend/.venv/bin/python"
PORT="${SORTOPS_PORT:-8200}"
UNIT="$HOME/.config/systemd/user/sortops.service"
NPU_LIB=/snap/intel-npu-driver/current/usr/lib/x86_64-linux-gnu
export PATH="$HOME/.local/node/bin:$PATH"

build() {
  (cd "$HERE/web" && npm ci --no-audit --no-fund && npm run build)
}

install() {
  [ -x "$PY" ] || { echo "Physical AI Studio venv not found at $PY (set STUDIO_DIR to your Physical AI Studio checkout)"; exit 1; }
  [ -f "$HERE/web/dist/index.html" ] || build
  mkdir -p "$(dirname "$UNIT")" "$HERE/data"
  cat > "$UNIT" <<UNIT
[Unit]
Description=SortOps sorting cell (http://127.0.0.1:$PORT)

[Service]
WorkingDirectory=$HERE/backend
Environment=PYTHONPATH=$HERE/backend PYTHONUNBUFFERED=1 SORTOPS_PORT=$PORT SORTOPS_DATA=$HERE/data
Environment=LD_LIBRARY_PATH=$NPU_LIB
ExecStart=$PY -m sortops
Restart=on-failure
RestartSec=3

[Install]
WantedBy=default.target
UNIT
  # First start on the PC drives the real arm and cameras (override in data/config.json).
  [ -f "$HERE/data/config.json" ] || echo '{"hardware": {"backend": "real"}}' > "$HERE/data/config.json"
  systemctl --user daemon-reload
  echo "Installed. Start with: $0 start   (stop Physical AI Studio sessions using the arm/cameras first)"
}

case "${1:-status}" in
  install) install ;;
  build) build ;;
  start|stop|restart|status) systemctl --user "$1" sortops --no-pager ;;
  logs) journalctl --user -u sortops -n "${2:-80}" --no-pager ;;
  open) WAYLAND_DISPLAY="${WAYLAND_DISPLAY:-wayland-0}" xdg-open "http://127.0.0.1:$PORT" >/dev/null 2>&1 & ;;
  foreground)
    cd "$HERE/backend"
    LD_LIBRARY_PATH="$NPU_LIB" PYTHONPATH="$HERE/backend" SORTOPS_DATA="$HERE/data" SORTOPS_PORT="$PORT" exec "$PY" -m sortops ;;
  *) echo "usage: $0 install|start|stop|restart|status|logs|open|foreground|build"; exit 1 ;;
esac
