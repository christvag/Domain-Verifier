import streamlit as st
import time
import os
import datetime
import pandas as pd
import re
import asyncio
import threading
from pathlib import Path

from database_manager import initialize_db, fetch_past_runs, clear_db

# --- Page and State Setup ---
st.set_page_config(page_title="Domain Verifier", layout="centered")
st.title("Domain Verifier")

# --- DB Initialization ---
initialize_db()

# --- Session State Initialization ---
for key in [
    "process_running",
    "process_finished",
    "verification_error",
    "last_upload_path",
    "reachable_df",
    "verification_complete_flag",
    "was_stopped_flag",
    "completion_timestamp",
]:
    if key not in st.session_state:
        st.session_state[key] = (
            False
            if key
            in [
                "process_running",
                "process_finished",
                "verification_complete_flag",
                "was_stopped_flag",
            ]
            else None
        )

# --- Global Variables and Placeholders ---
status_placeholder = st.empty()
LOG_FILE_PATH = os.path.join(os.getcwd(), "process.log")

# --- UI: Instructions and File Uploader ---
st.markdown(
    """
    **Instructions:**
    1.  Upload your domain list as a CSV file.
    2.  Click **Start** to begin. The progress box will automatically update.
    3.  When complete, the app will stop refreshing, and the Start button will be re-enabled.
    """
)
uploaded_file = st.file_uploader("Upload your domain CSV file", type=["csv"])
num_times = st.number_input(
    "How many times to process the file?", min_value=1, max_value=100, value=1
)


# --- Backend and Helper Functions ---
def get_latest_progress():
    """Reads the last known progress from the log file."""
    try:
        if not os.path.exists(LOG_FILE_PATH):
            return None
        with open(LOG_FILE_PATH, "r", encoding="utf-8", errors="ignore") as f:
            for line in reversed(list(f)):
                if line.startswith("PROGRESS|"):
                    match = re.match(
                        r"PROGRESS\|pass=(?P<pass>\w+)\|current=(?P<current>\d+)\|total=(?P<total>\d+)\|percent=(?P<percent>\d+)\|stage=(?P<stage>.+)",
                        line.strip(),
                    )
                    if match:
                        return match.groupdict()
    except Exception:
        return None
    return None


def check_if_stopped():
    """Checks if the log file indicates a manual stop."""
    try:
        if not os.path.exists(LOG_FILE_PATH):
            return False
        with open(LOG_FILE_PATH, "r", encoding="utf-8", errors="ignore") as f:
            return "PROCESS_STOPPED_BY_USER" in f.read()
    except Exception:
        return False


def run_verification_in_thread(file_path, retries, column_name):
    """
    Wrapper function to run the asyncio domain verification in a thread.
    This thread's ONLY job is to do work and set a completion flag.
    """

    def thread_target():
        completion_file = Path("verification_complete.flag")
        try:
            print(f"[DEBUG] Background thread starting verification...")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            from verify_websites import verify_domains

            start_time = time.time()
            reachable_df, _ = loop.run_until_complete(
                verify_domains(
                    file_path,
                    start_time=start_time,
                    retries=retries,
                    column=column_name,
                    log_path=LOG_FILE_PATH,
                )
            )
            st.session_state.reachable_df = reachable_df
            st.session_state.verification_error = None
            print(
                f"[DEBUG] Background thread completed successfully. Setting completion flag..."
            )
        except Exception as e:
            st.session_state.verification_error = str(e)
            print(f"[DEBUG] Error in background thread: {e}")
        finally:
            st.session_state.verification_complete_flag = True
            st.session_state.completion_timestamp = time.time()
            completion_file.touch()  # Create file signal
            print(
                f"[DEBUG] Background thread finished. Completion flag set at {st.session_state.completion_timestamp}"
            )
            print(f"[DEBUG] Created completion file: {completion_file}")

    # Clean up any existing completion file
    completion_file = Path("verification_complete.flag")
    if completion_file.exists():
        completion_file.unlink()

    t = threading.Thread(target=thread_target, daemon=True)
    t.start()
    print(f"[DEBUG] Started background thread for verification")


def display_progress_and_logs():
    """A regular function to display the progress and logs UI."""
    st.markdown("---")
    progress_info = get_latest_progress()

    if progress_info and st.session_state.get("process_running"):
        stage = progress_info.get("stage", "...")
        st.markdown(f"**Current Stage:** `{stage}`")

    if progress_info:
        percent = int(progress_info.get("percent", 0))
        st.progress(percent / 100, text=f"{percent}% Complete")
    else:
        if not st.session_state.get("process_finished"):
            st.info("Start a process to see live updates.")

    if os.path.exists(LOG_FILE_PATH):
        log_container = st.container(height=300)
        with open(LOG_FILE_PATH, "r", encoding="utf-8", errors="ignore") as f:
            log_lines = f.readlines()
        log_container.code("".join(reversed(log_lines)), language="log")


@st.fragment(run_every=0.5)
def show_running_ui():
    """Fragment that runs periodically to update the UI."""
    if st.session_state.get("process_running"):
        completion_file = Path("verification_complete.flag")
        if (
            st.session_state.get("verification_complete_flag")
            or completion_file.exists()
        ):
            # This is the key: we check for completion *inside* the fragment's run.
            print(f"[DEBUG] Fragment detected completion flag. Updating UI state...")
            st.session_state.process_running = False
            st.session_state.process_finished = True
            st.session_state.verification_complete_flag = False
            st.session_state.was_stopped_flag = check_if_stopped()
            # Clean up the completion file
            if completion_file.exists():
                completion_file.unlink()
                print(f"[DEBUG] Cleaned up completion file")
            print(
                f"[DEBUG] UI state updated. process_running=False, process_finished=True"
            )
            print(f"[DEBUG] Triggering full app rerun to update Start button...")
            st.rerun()  # Changed from st.rerun(scope="fragment") to full app rerun
        else:
            # While running, just update the logs
            display_progress_and_logs()
    else:
        # If not running, this fragment should do nothing.
        # This prevents the "does not exist" error.
        pass


# --- Main Application Logic ---
# Check for completion flag right at the top of the script logic
completion_file = Path("verification_complete.flag")
if st.session_state.get("verification_complete_flag") or completion_file.exists():
    print(f"[DEBUG] Main logic detected completion flag. Updating UI state...")
    st.session_state.process_running = False
    st.session_state.process_finished = True
    st.session_state.verification_complete_flag = False
    st.session_state.was_stopped_flag = check_if_stopped()
    # Clean up the completion file
    if completion_file.exists():
        completion_file.unlink()
        print(f"[DEBUG] Main logic cleaned up completion file")
    print(
        f"[DEBUG] Main logic updated UI state. process_running=False, process_finished=True"
    )
    print(f"[DEBUG] Main logic triggering full app rerun...")
    st.rerun()  # Changed from st.rerun(scope="fragment") to full app rerun


upload_path = None
if uploaded_file:
    upload_dir = "upload"
    os.makedirs(upload_dir, exist_ok=True)
    upload_path = os.path.join(upload_dir, uploaded_file.name)
    with open(upload_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.session_state.last_upload_path = upload_path

if st.button(
    "Start Verification",
    use_container_width=True,
    disabled=st.session_state.process_running,
):
    if upload_path:
        print(f"[DEBUG] Start button clicked. Initializing verification process...")
        st.session_state.process_running = True
        st.session_state.process_finished = False
        st.session_state.verification_complete_flag = False
        st.session_state.was_stopped_flag = False
        st.session_state.reachable_df = None
        if os.path.exists(LOG_FILE_PATH):
            os.remove(LOG_FILE_PATH)
        run_verification_in_thread(upload_path, num_times, "website")
        print(f"[DEBUG] Verification process initialized. process_running=True")
    else:
        st.warning("Please upload a CSV file first.")

# --- UI Display Logic ---
if st.session_state.process_running:
    status_placeholder.info("⚙️ Processing... See live progress below.")
    if st.button("Stop Verification", use_container_width=True, type="primary"):
        Path("stop_processing.flag").touch()
        status_placeholder.warning(
            "🛑 Stop signal sent. The process will halt gracefully after the current batch. Please wait..."
        )
        st.session_state.was_stopped_flag = True

    show_running_ui()

elif st.session_state.process_finished:
    if st.session_state.get("was_stopped_flag"):
        status_placeholder.warning(
            "🛑 Process stopped by user. Ready to resume or start a new run."
        )
    else:
        status_placeholder.success("✅ Verification complete! Ready for next run.")
    st.markdown("### Final Status")
    display_progress_and_logs()

    if st.session_state.reachable_df is not None:
        df = st.session_state.reachable_df
        csv_data = df.to_csv(index=False).encode("utf-8")
        base_name = os.path.splitext(
            os.path.basename(st.session_state.last_upload_path)
        )[0]
        st.download_button(
            label="⬇️ Download Result",
            data=csv_data,
            file_name=f"{base_name}_verified.csv",
            use_container_width=True,
        )
else:
    # Initial state
    st.markdown("---")
    st.info("Start a process to see live updates.")


# --- Display Past Runs ---
@st.fragment
def show_past_runs():
    """Displays a table of past runs with a manual refresh button."""
    st.markdown("---")
    st.subheader("Past Runs")

    refresh_button, clear_button = st.columns(2)
    with refresh_button:
        if st.button("🔄 Refresh History"):
            st.rerun(scope="fragment")
    with clear_button:
        if st.button("🧹 Clear History", on_click=clear_db):
            st.rerun(scope="fragment")

    past_runs = fetch_past_runs()

    if not past_runs:
        st.info("No past runs found. Complete a verification to see history here.")
    else:
        cols = st.columns((2, 1, 1, 1, 1, 1))
        headers = [
            "File Processed",
            "Processing Time",
            "Reachable",
            "Unreachable",
            "Reachable File",
            "Full Report",
        ]
        for col, header in zip(cols, headers):
            col.write(f"**{header}**")

        for run in past_runs:
            cols = st.columns((2, 1, 1, 1, 1, 1))
            cols[0].write(run["filename"])
            cols[1].write(f"{run['processing_time']:.2f}s")
            cols[2].write(f"✔️ {run['reachable_count']}")
            cols[3].write(f"❌ {run['unreachable_count']}")

            with open(run["reachable_filepath"], "rb") as f:
                cols[4].download_button(
                    "⬇️",
                    f.read(),
                    file_name=Path(run["reachable_filepath"]).name,
                    key=f"reachable_{run['id']}",
                )

            with open(run["report_filepath"], "rb") as f:
                cols[5].download_button(
                    "⬇️",
                    f.read(),
                    file_name=Path(run["report_filepath"]).name,
                    key=f"report_{run['id']}",
                )


show_past_runs()
