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

import threading
import time
from collections import deque
from typing import Optional, List, Dict, Any
from enum import Enum
from dataclasses import dataclass
from datetime import datetime


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


class CentralizedLogger:
    """
    Thread-safe centralized logger for the Domain Verifier application.

    Features:
    - In-memory log storage with circular buffer
    - Progress tracking with structured context
    - Thread-safe operations
    - Real-time log retrieval
    - Integration with Streamlit session state
    """

    def __init__(self, max_entries: int = 1000):
        """
        Initialize the centralized logger.

        Args:
            max_entries: Maximum number of log entries to keep in memory
        """
        self.max_entries = max_entries
        self._logs = deque(maxlen=max_entries)
        self._lock = threading.RLock()
        self._progress_info = {}

    def _add_entry(
        self, level: LogLevel, message: str, context: Optional[Dict[str, Any]] = None
    ):
        """Thread-safe method to add a log entry"""
        entry = LogEntry(
            timestamp=time.time(), level=level, message=message, context=context or {}
        )

        with self._lock:
            self._logs.append(entry)

            # Update progress info if this is a progress entry
            if level == LogLevel.PROGRESS and context:
                self._progress_info.update(context)

    def debug(self, message: str, **context):
        """Log a debug message"""
        self._add_entry(LogLevel.DEBUG, message, context)

    def info(self, message: str, **context):
        """Log an info message"""
        self._add_entry(LogLevel.INFO, message, context)

    def warning(self, message: str, **context):
        """Log a warning message"""
        self._add_entry(LogLevel.WARNING, message, context)

    def error(self, message: str, **context):
        """Log an error message"""
        self._add_entry(LogLevel.ERROR, message, context)

    def progress(self, message: str, **context):
        """
        Log a progress update with structured context.

        Expected context keys:
        - pass: Current pass (e.g., "head", "get", "contact_crawl")
        - current: Current item number
        - total: Total items
        - percent: Completion percentage
        - stage: Current stage description
        """
        self._add_entry(LogLevel.PROGRESS, message, context)

    def get_logs(
        self, level_filter: Optional[LogLevel] = None, last_n: Optional[int] = None
    ) -> List[LogEntry]:
        """
        Get log entries with optional filtering.

        Args:
            level_filter: Only return logs of this level
            last_n: Only return the last N entries

        Returns:
            List of log entries
        """
        with self._lock:
            logs = list(self._logs)

        if level_filter:
            logs = [log for log in logs if log.level == level_filter]

        if last_n:
            logs = logs[-last_n:]

        return logs

    def get_logs_as_text(
        self, level_filter: Optional[LogLevel] = None, last_n: Optional[int] = None
    ) -> str:
        """Get logs formatted as text string"""
        logs = self.get_logs(level_filter, last_n)
        return "\n".join(str(log) for log in logs)

    def get_latest_progress(self) -> Optional[Dict[str, Any]]:
        """Get the latest progress information"""
        with self._lock:
            return self._progress_info.copy() if self._progress_info else None

    def clear(self):
        """Clear all logs and progress info"""
        with self._lock:
            self._logs.clear()
            self._progress_info.clear()

    def get_stats(self) -> Dict[str, int]:
        """Get logging statistics"""
        with self._lock:
            logs = list(self._logs)

        stats = {level.value: 0 for level in LogLevel}
        for log in logs:
            stats[log.level.value] += 1

        stats["total"] = len(logs)
        return stats


# Global logger instance
_global_logger: Optional[CentralizedLogger] = None
_logger_lock = threading.Lock()


def get_logger() -> CentralizedLogger:
    """Get the global logger instance (singleton pattern)"""
    global _global_logger

    if _global_logger is None:
        with _logger_lock:
            if _global_logger is None:
                _global_logger = CentralizedLogger()

    return _global_logger


def reset_logger():
    """Reset the global logger (useful for testing)"""
    global _global_logger
    with _logger_lock:
        _global_logger = None


# Convenience functions for direct logging
def debug(message: str, **context):
    """Log a debug message using the global logger"""
    get_logger().debug(message, **context)


def info(message: str, **context):
    """Log an info message using the global logger"""
    get_logger().info(message, **context)


def warning(message: str, **context):
    """Log a warning message using the global logger"""
    get_logger().warning(message, **context)


def error(message: str, **context):
    """Log an error message using the global logger"""
    get_logger().error(message, **context)


def progress(message: str, **context):
    """Log a progress update using the global logger"""
    get_logger().progress(message, **context)


# Example usage and testing
if __name__ == "__main__":
    import asyncio

    async def test_logger():
        """Test the centralized logger"""
        logger = get_logger()

        # Test basic logging
        logger.info("Starting test")
        logger.debug("Debug message", test_param="debug_value")
        logger.warning("Warning message", category="test")
        logger.error("Error message", error_code=500)

        # Test progress logging
        for i in range(5):
            logger.progress(
                f"Processing item {i+1}",
                pass_="test_pass",
                current=i + 1,
                total=5,
                percent=int((i + 1) / 5 * 100),
                stage=f"Testing item {i+1}",
            )
            await asyncio.sleep(0.1)

        # Display results
        print("=== All Logs ===")
        print(logger.get_logs_as_text())

        print("\n=== Progress Only ===")
        print(logger.get_logs_as_text(LogLevel.PROGRESS))

        print("\n=== Latest Progress ===")
        print(logger.get_latest_progress())

        print("\n=== Stats ===")
        print(logger.get_stats())

    # Run test
    asyncio.run(test_logger())
