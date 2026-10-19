"""GUI-side preview client. Standard library only.

The client owns a single long-lived worker process, keeps exactly one request
in flight, and coalesces rapid parameter changes down to the newest pending
request. Worker stdout is a JSON-lines protocol; anything else is logged.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path


class PreviewClient:
    def __init__(self, python: Path, project_dir: Path, temp_root: Path,
                 events: queue.Queue, name: str = "preview") -> None:
        self.python = Path(python)
        self.project_dir = Path(project_dir)
        self.temp_root = Path(temp_root)
        self.events = events
        self.name = name
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._reader: threading.Thread | None = None
        self._stderr_reader: threading.Thread | None = None
        self._in_flight: int | None = None
        self._pending: dict | None = None
        self._next_id = 0
        self._closed = False
        self._generation = 0

    # -- lifecycle ---------------------------------------------------------
    def _alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self) -> bool:
        if self._closed:
            return False
        if self._alive():
            return True
        if not self.python.exists():
            self.events.put((self.name, {"kind": "dead",
                "error": f"интерпретатор не найден: {self.python}"}))
            return False
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
        try:
            self._proc = subprocess.Popen(
                [str(self.python), "-u", "-m", "batchresize.preview_worker",
                 str(self.temp_root)],
                cwd=str(self.project_dir), env=env,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, encoding="utf-8",
                errors="replace", bufsize=1,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except OSError as exc:
            self.events.put((self.name, {"kind": "dead",
                "error": f"не удалось запустить preview worker: {exc}"}))
            return False
        self._generation += 1
        generation = self._generation
        self._reader = threading.Thread(target=self._read_stdout,
                                        args=(self._proc, generation),
                                        daemon=True)
        self._stderr_reader = threading.Thread(target=self._read_stderr,
                                               args=(self._proc, generation),
                                               daemon=True)
        self._reader.start()
        self._stderr_reader.start()
        return True

    def _read_stdout(self, proc: subprocess.Popen, generation: int) -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            text = line.strip()
            if not text:
                continue
            try:
                response = json.loads(text)
            except json.JSONDecodeError:
                self.events.put((self.name, {"kind": "log", "text": text}))
                continue
            self.events.put((self.name, response))
            with self._lock:
                if response.get("id") == self._in_flight:
                    self._in_flight = None
                pending, self._pending = self._pending, None
                if pending is not None and self._in_flight is None:
                    self._in_flight = pending.get("id")
            if pending is not None:
                self._send(pending)
        code = proc.wait()
        with self._lock:
            self._in_flight = None
            self._pending = None
        if generation == self._generation and not self._closed:
            self.events.put((self.name, {"kind": "dead",
                "error": f"preview worker завершился с кодом {code}"}))

    def _read_stderr(self, proc: subprocess.Popen, generation: int) -> None:
        assert proc.stderr is not None
        for line in proc.stderr:
            text = line.rstrip()
            if text:
                self.events.put((self.name, {"kind": "log", "text": text}))

    # -- requests ----------------------------------------------------------
    def request(self, payload: dict) -> int:
        with self._lock:
            self._next_id += 1
            request_id = self._next_id
            payload = {**payload, "id": request_id}
            if self._in_flight is not None:
                self._pending = payload
                return request_id
            self._in_flight = request_id
        if not self.start():
            with self._lock:
                self._in_flight = None
                self._pending = None
            return request_id
        self._send(payload)
        return request_id

    def invalidate(self, request_id: int) -> None:
        """Ignore the response for this id in the UI, but keep the worker busy."""
        self.events.put((self.name, {"kind": "invalidated", "id": request_id}))

    def _send(self, payload: dict) -> None:
        proc = self._proc
        if proc is None or proc.poll() is not None or proc.stdin is None:
            return
        line = json.dumps(payload, ensure_ascii=False) + "\n"
        try:
            with self._write_lock:
                proc.stdin.write(line)
                proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            with self._lock:
                self._in_flight = None
                self._pending = None
            self.events.put((self.name, {"kind": "dead",
                "error": f"preview worker недоступен: {exc}"}))

    def cancel(self) -> None:
        """Stop the worker (for example while a large batch owns the GPU)."""
        with self._lock:
            self._pending = None
            self._in_flight = None
        self._terminate()

    def close(self) -> None:
        self._closed = True
        with self._lock:
            self._pending = None
            self._in_flight = None
        proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                if proc.stdin is not None:
                    proc.stdin.write('{"cmd":"quit"}\n')
                    proc.stdin.flush()
                    proc.stdin.close()
            except (OSError, ValueError):
                pass
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=2)
        self._proc = None

    def _terminate(self) -> None:
        proc = self._proc
        if proc is None:
            return
        self._generation += 1  # silence the reader's death notice
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass
        self._proc = None
