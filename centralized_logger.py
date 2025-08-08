"""
Centralized Logger System

Thread-safe, in-memory logging system for the Domain Verifier application.
Provides real-time log updates without file I/O dependencies.

Features:
- Thread-safe log storage
- Real-time progress tracking
- Structured log levels
- Memory-efficient circular buffer
- Integration with Streamlit session state
"""

from __future__ import annotations
import threading
import time
from typing import Optional, List, Dict, Any
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
import logging


class LogLevel(Enum):
    """Log levels for categorizing messages"""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    PROGRESS = "PROGRESS"


@dataclass
class LogEntry:
    """Structured log entry"""

    timestamp: float
    level: LogLevel
    message: str
    context: Optional[Dict[str, Any]] = None

    def __str__(self) -> str:
        dt = datetime.fromtimestamp(self.timestamp)
        time_str = dt.strftime("%H:%M:%S.%f")[:-3]  # Millisecond precision

        if self.level == LogLevel.PROGRESS and self.context:
            return f"[{time_str}] PROGRESS | {self.context.get('stage', 'Unknown')}: {self.message}"
        else:
            return f"[{time_str}] {self.level.value}: {self.message}"


class _CentralLogger:
    """
    Thread-safe centralized logger for the Domain Verifier application.

    Features:
    - In-memory log storage with circular buffer
    - Progress tracking with structured context
    - Thread-safe operations
    - Real-time log retrieval
    - Integration with Streamlit session state
    """

    def __init__(self, name="contact_form_crawler", max_entries: int = 1000):
        """
        Initialize the centralized logger.

        Args:
            name: Logger name
            max_entries: Maximum number of log entries to keep in memory
        """
        self.name = name
        self.max_entries = max_entries
        self._logs: List[str] = []
        self._latest_progress = {"stage": "", "percent": 0}
        self._lock = threading.RLock()

        # Initialize Python logger
        self._logger = logging.getLogger(name)
        if not self._logger.hasHandlers():
            handler = logging.StreamHandler()
            # Only print the message, no timestamp or log level
            formatter = logging.Formatter("%(message)s")
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)
        self._logger.setLevel(logging.INFO)

    def log(self, msg: str):
        """Thread-safe method to add a log message"""
        with self._lock:
            self._logs.append(msg)

    def info(self, msg: str):
        """Log an informational message"""
        self.log(f"[INFO] {msg}")

    def warning(self, msg: str):
        """Log a warning message"""
        self.log(f"[WARNING] {msg}")

    def error(self, msg: str):
        """Log an error message"""
        self.log(f"[ERROR] {msg}")

    def progress(self, stage: str, percent: int):
        """
        Log a progress update with structured context.

        Expected context keys:
        - pass: Current pass (e.g., "head", "get", "contact_crawl")
        - current: Current item number
        - total: Total items
        - percent: Completion percentage
        - stage: Current stage description
        """
        with self._lock:
            self._latest_progress = {"stage": stage, "percent": percent}
            self._logs.append(f"[{percent}%] {stage}")

    def get_latest_progress(self):
        """Get the latest progress information"""
        with self._lock:
            return dict(self._latest_progress)

    def get_logs_as_text(self, last_n: Optional[int] = None) -> str:
        """Get logs formatted as text string"""
        with self._lock:
            logs = self._logs[-last_n:] if last_n else self._logs[:]
            return "\n".join(logs)

    def clear(self):
        """Clear all logs and progress info"""
        with self._lock:
            self._logs.clear()
            self._latest_progress = {"stage": "", "percent": 0}


_singleton: Optional[_CentralLogger] = None


def get_logger() -> _CentralLogger:
    """Get the global logger instance (singleton pattern)"""
    global _singleton
    if _singleton is None:
        _singleton = _CentralLogger()
    return _singleton


# Example usage and testing
if __name__ == "__main__":
    import asyncio

    async def test_logger():
        """Test the centralized logger"""
        logger = get_logger()

        # Test basic logging
        logger.log("Starting test")
        logger.log("Debug message: test_param=debug_value")
        logger.log("Warning message: category=test")
        logger.log("Error message: error_code=500")

        # Test progress logging
        for i in range(5):
            logger.progress(
                stage=f"Testing item {i+1}",
                percent=int((i + 1) / 5 * 100),
            )
            await asyncio.sleep(0.1)

        # Display results
        print("=== All Logs ===")
        print(logger.get_logs_as_text())

        print("\n=== Progress Only ===")
        print(logger.get_logs_as_text(LogLevel.PROGRESS))

        print("\n=== Latest Progress ===")
        print(logger.get_latest_progress())

    # Run test
    asyncio.run(test_logger())
