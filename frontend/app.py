from __future__ import annotations

import os

from phoenix_frontend import create_app

app = create_app()


if __name__ == "__main__":
    host = str(os.environ.get("PHOENIX_UI_HOST", "127.0.0.1"))
    port = int(os.environ.get("PHOENIX_UI_PORT", "5050"))
    debug = str(os.environ.get("PHOENIX_UI_DEBUG", "true")).strip().lower() in {"1", "true", "yes", "y"}
    app.run(host=host, port=port, debug=debug)
