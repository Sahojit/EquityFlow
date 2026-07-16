"""Structured logging configuration for AlphaAgents.

Call ``configure_logging()`` once at application startup (in api/main.py lifespan
or at the top of any CLI entry point). After that, every module's
``logging.getLogger(__name__)`` will emit JSON-formatted log records to both
the console and a rotating file (logs/alpha_agents.log).

Log levels by module:
  DEBUG  — llm.*         (token counts, raw LLM responses)
  INFO   — agents.*      (per-node start/end, key metrics)
  INFO   — api.*         (request/response lifecycle)
  WARNING — root         (everything else defaults to WARNING)
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
from datetime import UTC, datetime


class _JsonFormatter(logging.Formatter):
    """Emit each log record as a single-line JSON object.

    Fields: timestamp, level, logger, message, and any extra kwargs
    passed to the logger call.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        for key, val in record.__dict__.items():
            if key not in {
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "exc_info", "exc_text", "stack_info",
                "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process", "message",
                "taskName",
            }:
                try:
                    json.dumps(val)
                    payload[key] = val
                except (TypeError, ValueError):
                    pass
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(
    *,
    log_dir: str = "logs",
    log_file: str = "alpha_agents.log",
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> None:
    """Configure JSON structured logging for the whole application.

    Sets per-module log levels and attaches two handlers:
      - StreamHandler (console) — human-readable JSON, coloured by level
      - RotatingFileHandler (file) — full JSON, rotated at 10 MB

    Call once at startup; calling again is a no-op (handlers already attached).

    Args:
        log_dir: Directory for the log file. Created if it does not exist.
        log_file: Log file name inside log_dir.
        max_bytes: Maximum size of each log file before rotation.
        backup_count: Number of rotated files to retain.
    """
    root = logging.getLogger()
    if root.handlers:
        return

    os.makedirs(log_dir, exist_ok=True)

    json_fmt = _JsonFormatter()

    console = logging.StreamHandler()
    console.setFormatter(json_fmt)
    console.setLevel(logging.DEBUG)

    file_handler = logging.handlers.RotatingFileHandler(
        filename=os.path.join(log_dir, log_file),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(json_fmt)
    file_handler.setLevel(logging.DEBUG)

    root.setLevel(logging.WARNING)
    root.addHandler(console)
    root.addHandler(file_handler)

    logging.getLogger("llm").setLevel(logging.DEBUG)
    logging.getLogger("agents").setLevel(logging.INFO)
    logging.getLogger("api").setLevel(logging.INFO)
    logging.getLogger("graph").setLevel(logging.INFO)
