# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Satellite Environment.

This module creates an HTTP server that exposes the SatelliteEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

import importlib

# 1) Try absolute imports first (this handles the Docker layout where /app/env
#    is on PYTHONPATH and modules like `models` and `server` are top-level).
try:
    from models import SatelliteAction, SatelliteObservation
    from server.satellite_environment import SatelliteEnvironment
except Exception:  # pragma: no cover - tolerate multiple packaging layouts
    # 2) Try package-relative imports (when running as `satellite.server.app`).
    try:
        from ..models import SatelliteAction, SatelliteObservation
        from .satellite_environment import SatelliteEnvironment
    except Exception:
        # 3) Try common installed package names (e.g., `satellite` or `env`).
        SatelliteAction = None
        SatelliteObservation = None
        SatelliteEnvironment = None
        for pkg in ("satellite", "env"):
            try:
                mod_models = importlib.import_module(f"{pkg}.models")
                mod_server_env = importlib.import_module(f"{pkg}.server.satellite_environment")
                SatelliteAction = getattr(mod_models, "SatelliteAction")
                SatelliteObservation = getattr(mod_models, "SatelliteObservation")
                SatelliteEnvironment = getattr(mod_server_env, "SatelliteEnvironment")
                break
            except Exception:
                continue

        if SatelliteAction is None:
            # 4) Last resort: file-based fallback (load sibling files directly).
            try:
                import importlib.util
                from pathlib import Path

                here = Path(__file__).resolve().parent
                pkg_root = here.parent
                models_path = pkg_root / "models.py"
                server_env_path = here / "satellite_environment.py"

                if models_path.exists() and server_env_path.exists():
                    spec_models = importlib.util.spec_from_file_location(
                        "satellite_models_fallback", str(models_path)
                    )
                    mod_models = importlib.util.module_from_spec(spec_models)  # type: ignore[arg-type]
                    spec_models.loader.exec_module(mod_models)  # type: ignore[attr-defined]

                    spec_server_env = importlib.util.spec_from_file_location(
                        "satellite_server_env_fallback", str(server_env_path)
                    )
                    mod_server_env = importlib.util.module_from_spec(spec_server_env)  # type: ignore[arg-type]
                    spec_server_env.loader.exec_module(mod_server_env)  # type: ignore[attr-defined]

                    SatelliteAction = getattr(mod_models, "SatelliteAction")
                    SatelliteObservation = getattr(mod_models, "SatelliteObservation")
                    SatelliteEnvironment = getattr(mod_server_env, "SatelliteEnvironment")
                else:
                    raise
            except Exception:
                raise


# Create the app with web interface and README integration
app = create_app(
    SatelliteEnvironment,
    SatelliteAction,
    SatelliteObservation,
    env_name="satellite",
    max_concurrent_envs=1,  # increase this number to allow more concurrent WebSocket sessions
)


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m satellite.server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn satellite.server.app:app --workers 4
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


def main():
    """CLI-compatible main entry point for OpenEnv validation and local runs."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    run_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
