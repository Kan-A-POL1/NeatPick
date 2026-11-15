"""``python -m sortops`` — serve the cell API and UI (``SORTOPS_HOST``/``SORTOPS_PORT``, default 127.0.0.1:8200)."""

from __future__ import annotations

import os

import uvicorn

from sortops.server import create_app


def main() -> None:
    uvicorn.run(
        create_app(),
        host=os.environ.get("SORTOPS_HOST", "127.0.0.1"),
        port=int(os.environ.get("SORTOPS_PORT", "8200")),
        log_level="warning",
    )


if __name__ == "__main__":
    main()
