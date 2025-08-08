import sqlite3
import shutil
from pathlib import Path

DB_PATH = "verifier.db"


def initialize_db():
    """Initializes the SQLite database and creates the necessary tables if they don't exist."""
    Path(DB_PATH).touch(exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        # Verification runs
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                processing_time REAL NOT NULL,
                reachable_count INTEGER NOT NULL,
                unreachable_count INTEGER NOT NULL,
                reachable_filepath TEXT NOT NULL,
                report_filepath TEXT NOT NULL,
                completion_timestamp REAL,
                was_stopped_flag INTEGER,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        # Try to add missing columns if table already exists
        try:
            cur.execute("ALTER TABLE runs ADD COLUMN completion_timestamp REAL")
        except Exception:
            pass
        try:
            cur.execute("ALTER TABLE runs ADD COLUMN was_stopped_flag INTEGER")
        except Exception:
            pass

        # Process runs (contact discovery)
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS process_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                original_filename TEXT NOT NULL,
                reachable_filepath TEXT NOT NULL,
                contact_forms_filepath TEXT NOT NULL,
                success_rate REAL NOT NULL,
                processing_time REAL NOT NULL,
                metrics_json_filepath TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        try:
            cur.execute("ALTER TABLE process_runs ADD COLUMN metrics_json_filepath TEXT")
        except Exception:
            pass
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
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO runs (filename, processing_time, reachable_count, unreachable_count, reachable_filepath, report_filepath)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (filename, processing_time, reachable_count, unreachable_count, str(reachable_filepath), str(report_filepath)),
        )
        conn.commit()


def save_process_run_to_db(
    original_filename: str,
    reachable_filepath: str,
    contact_forms_filepath: str,
    success_rate: float,
    processing_time: float,
    metrics_json_filepath: str | None = None,
):
    """Saves the details of a contact form processing run to the database.
    
    Args:
        original_filename (str): Name of the original file uploaded
        reachable_filepath (str): Path to the CSV with reachable websites
        contact_forms_filepath (str): Path to the CSV with found contact forms
        success_rate (float): Percentage of reachable domains with contact forms
        processing_time (float): Time taken for the whole process in seconds
    """
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO process_runs (original_filename, reachable_filepath, contact_forms_filepath, success_rate, processing_time, metrics_json_filepath)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                original_filename,
                str(reachable_filepath),
                str(contact_forms_filepath),
                float(success_rate),
                float(processing_time),
                str(metrics_json_filepath) if metrics_json_filepath else None,
            ),
        )
        conn.commit()


def fetch_past_runs():
    """Fetches all past verification runs from the database."""
    initialize_db()  # Ensure DB exists when fetching
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT * FROM runs ORDER BY timestamp DESC")
        return cur.fetchall()


def fetch_process_runs():
    """Fetches all past contact form processing runs from the database."""
    initialize_db()  # Ensure DB exists when fetching
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT * FROM process_runs ORDER BY timestamp DESC")
        return cur.fetchall()


def clear_db():
    """Clears the database and removes all files from all, reachable, and upload folders."""
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM runs")
        cur.execute("DELETE FROM process_runs")
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
