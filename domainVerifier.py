import streamlit as st
import time
import os
import datetime
import pandas as pd
import re
import asyncio
import threading

# --- Page and State Setup ---
st.set_page_config(page_title="Domain Verifier", layout="centered")
st.title("Domain Verifier")

# --- Session State Initialization ---
# Initialize all keys to prevent errors on the first run
for key in [
    'process_running', 'process_finished', 'verification_error',
    'last_upload_path', 'reachable_df', 'verification_complete_flag'
]:
    if key not in st.session_state:
        st.session_state[key] = False if key in ['process_running', 'process_finished', 'verification_complete_flag'] else None

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
num_times = st.number_input("How many times to process the file?", min_value=1, max_value=100, value=1)

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

def run_verification_in_thread(file_path, retries, column_name):
    """
    Wrapper function to run the asyncio domain verification in a thread.
    This thread's ONLY job is to do work and set a completion flag.
    """
    def thread_target():
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            from verify_websites import verify_domains
            reachable_df, _ = loop.run_until_complete(
                verify_domains(file_path, retries=retries, column=column_name, log_path=LOG_FILE_PATH)
            )
            st.session_state.reachable_df = reachable_df
            st.session_state.verification_error = None
        except Exception as e:
            st.session_state.verification_error = str(e)
            print(f"Error in background thread: {e}")
        finally:
            st.session_state.verification_complete_flag = True

    t = threading.Thread(target=thread_target, daemon=True)
    t.start()
    # Wait for the thread to finish if needed, or let it run in the background.
    t.join()

# --- Dynamic Fragment Definition ---
refresh_interval = 0.5 if st.session_state.get('process_running') else None

@st.fragment(run_every=refresh_interval)
def show_progress_box():
    """
    This fragment checks for the completion flag and updates the state.
    It does NOT call st.rerun() itself, preventing race conditions.
    """
    if st.session_state.get('verification_complete_flag'):
        # If the flag is set, update the state.
        # The polling will stop naturally on the next script run
        # because 'refresh_interval' will become None.
        st.session_state.process_running = False
        st.session_state.process_finished = True
        st.session_state.verification_complete_flag = False

    # The rest of the function is for displaying progress as before.
    st.markdown("---")
    progress_info = get_latest_progress()

    if st.session_state.get('process_running'):
        st.markdown("### Live Progress (Refreshes automatically)")
    elif st.session_state.get('process_finished'):
        st.markdown("### Final Status")

    if progress_info:
        percent = int(progress_info.get("percent", 0))
        st.progress(percent / 100, text=f"{percent}% Complete")
    else:
        # Avoid showing "No progress" when we just finished.
        if not st.session_state.get('process_finished'):
            st.info("Start a process to see live updates.")

# --- Main Application Logic ---
upload_path = None
if uploaded_file:
    upload_dir = "upload"
    os.makedirs(upload_dir, exist_ok=True)
    upload_path = os.path.join(upload_dir, uploaded_file.name)
    with open(upload_path, "wb") as f: f.write(uploaded_file.getbuffer())
    st.session_state.last_upload_path = upload_path

if st.button("Start Verification", use_container_width=True, disabled=st.session_state.process_running):
    if upload_path:
        st.session_state.process_running = True
        st.session_state.process_finished = False
        st.session_state.verification_complete_flag = False
        st.session_state.reachable_df = None
        if os.path.exists(LOG_FILE_PATH): os.remove(LOG_FILE_PATH)
        run_verification_in_thread(upload_path, num_times, "website")
        st.rerun()
    else:
        st.warning("Please upload a CSV file first.")

# --- UI Display & Fragment Call ---
if st.session_state.process_running:
    status_placeholder.info("⚙️ Processing... See live progress below.")
elif st.session_state.process_finished:
    if st.session_state.verification_error:
        status_placeholder.error(f"An error occurred: {st.session_state.verification_error}")
    else:
        status_placeholder.success("✅ Verification complete! Ready for next run.")
    if st.session_state.reachable_df is not None:
        df = st.session_state.reachable_df
        csv_data = df.to_csv(index=False).encode('utf-8')
        base_name = os.path.splitext(os.path.basename(st.session_state.last_upload_path))[0]
        st.download_button(
            label="⬇️ Download Result", data=csv_data,
            file_name=f"{base_name}_verified.csv", use_container_width=True
        )

# This single call correctly handles all states.
show_progress_box()