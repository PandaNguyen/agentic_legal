from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
DEFAULT_LOG_FILE = "log.log"


def setup_logging(*, log_level: str = "INFO", log_file: str = DEFAULT_LOG_FILE) -> Path:
    level = getattr(logging, log_level.upper(), logging.INFO)
    log_path = Path(log_file)
    if not log_path.is_absolute():
        log_path = Path.cwd() / log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    file_handler = _get_existing_file_handler(root_logger, log_path)
    if file_handler is None:
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=10 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.set_name("agentic_legal_file")
        root_logger.addHandler(file_handler)

    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"):
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)

    return log_path


def _get_existing_file_handler(logger: logging.Logger, log_path: Path) -> RotatingFileHandler | None:
    resolved_log_path = log_path.resolve()
    for handler in logger.handlers:
        if not isinstance(handler, RotatingFileHandler):
            continue
        handler_path = Path(handler.baseFilename).resolve()
        if handler_path == resolved_log_path:
            return handler
    return None
