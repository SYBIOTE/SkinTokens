"""
Supervised bpy_server subprocess for RunPod / Docker.

uvicorn remains PID 1; bpy_server is a child process started from FastAPI
lifespan, health-checked via /ping, and torn down on shutdown.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time

import requests

logger = logging.getLogger("skintokens.bpy_supervisor")

BPY_PORT = int(os.environ.get("SKINTOKENS_BPY_PORT", "59876"))
BPY_PING_URL = f"http://127.0.0.1:{BPY_PORT}/ping"
STARTUP_TIMEOUT_S = int(os.environ.get("SKINTOKENS_BPY_STARTUP_TIMEOUT", "60"))


class BpySupervisor:
    """Start and monitor the bpy_server sidecar subprocess."""

    def __init__(self) -> None:
        self._proc: subprocess.Popen[bytes] | None = None

    def start(self) -> None:
        if self.is_healthy():
            return

        if self._proc is not None and self._proc.poll() is None:
            self._wait_ready()
            return

        app_dir = os.environ.get("SKINTOKENS_APP_DIR", "/app")
        logger.info("Starting bpy_server subprocess (port=%s, cwd=%s)", BPY_PORT, app_dir)
        self._proc = subprocess.Popen(
            [sys.executable, "bpy_server.py"],
            cwd=app_dir,
        )
        self._wait_ready()

    def _wait_ready(self) -> None:
        deadline = time.monotonic() + STARTUP_TIMEOUT_S
        while time.monotonic() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                code = self._proc.returncode
                raise RuntimeError(f"bpy_server exited before becoming ready (code={code})")
            if self._ping():
                logger.info("bpy_server is ready on port %s (pid=%s)", BPY_PORT, self._proc.pid if self._proc else "?")
                return
            time.sleep(0.5)
        raise RuntimeError(f"bpy_server failed to start within {STARTUP_TIMEOUT_S}s")

    def _ping(self) -> bool:
        try:
            response = requests.get(BPY_PING_URL, timeout=2)
            return response.status_code == 200 and response.text == "pong"
        except requests.RequestException:
            return False

    def is_healthy(self) -> bool:
        if self._proc is None or self._proc.poll() is not None:
            return False
        return self._ping()

    def stop(self) -> None:
        if self._proc is None:
            return
        if self._proc.poll() is None:
            logger.info("Stopping bpy_server (pid=%s)", self._proc.pid)
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
        self._proc = None


_supervisor = BpySupervisor()


def get_bpy_supervisor() -> BpySupervisor:
    return _supervisor
