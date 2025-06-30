import sqlite3
import shutil
import os
from pathlib import Path

DB_PATH = "verifier.db"


def initialize_db():
    """Initializes the SQLite database and creates the 'runs' table if it doesn't exist."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                processing_time REAL NOT NULL,
                reachable_count INTEGER NOT NULL,
                unreachable_count INTEGER NOT NULL,
                reachable_filepath TEXT NOT NULL,
                report_filepath TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()


def save_run_to_db(
    filename,
    processing_time,
    reachable_count,
    unreachable_count,
    reachable_filepath,
    report_filepath,
):
    """Saves the details of a verification run to the database."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO runs (filename, processing_time, reachable_count, unreachable_count, reachable_filepath, report_filepath)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                filename,
                processing_time,
                reachable_count,
                unreachable_count,
                str(reachable_filepath),
                str(report_filepath),
            ),
        )
        conn.commit()


def fetch_past_runs():
    """Fetches all past verification runs from the database."""
    initialize_db()  # Ensure DB exists when fetching
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM runs ORDER BY timestamp DESC")
        return cursor.fetchall()


def clear_db():
    """Clears the database and removes all files from all, reachable, and upload folders."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM runs")
        conn.commit()
        print("Database cleared")

    # Define directories to clear
    directories_to_clear = ["all", "reachable", "upload"]

    for directory in directories_to_clear:
        dir_path = Path(directory)
        if dir_path.exists() and dir_path.is_dir():
            try:
                # Remove all files in the directory
                for file_path in dir_path.iterdir():
                    if file_path.is_file():
                        file_path.unlink()
                        print(f"Removed file: {file_path}")
                    elif file_path.is_dir():
                        shutil.rmtree(file_path)
                        print(f"Removed directory: {file_path}")
                print(f"Cleared all contents from {directory}/ folder")
            except Exception as e:
                print(f"Error clearing {directory}/ folder: {e}")
        else:
            print(f"{directory}/ folder does not exist or is not a directory")
