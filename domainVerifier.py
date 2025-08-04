import streamlit as st
import time
import os
import pandas as pd
import asyncio
import threading
from pathlib import Path

from database_manager import initialize_db, fetch_past_runs, clear_db
from contact_form_crawler import ContactFormCrawler
from centralized_logger import get_logger

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
    "contact_crawl_running",
    "contact_crawl_results",
    "enable_contact_crawling",
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
                "contact_crawl_running",
                "enable_contact_crawling",
            ]
            else None
        )

# --- Global Variables and Placeholders ---
status_placeholder = st.empty()
logger = get_logger()

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

# Contact form crawler option
st.session_state.enable_contact_crawling = st.checkbox(
    "🔍 Enable contact form discovery on reachable domains",
    value=st.session_state.enable_contact_crawling,
    help="After domain verification, automatically search for contact forms on reachable websites",
)


# --- Backend and Helper Functions ---
def get_latest_progress():
    """Gets the latest progress from the centralized logger."""
    return logger.get_latest_progress()


def check_if_stopped():
    """Checks if the logger indicates a manual stop."""
    logs = logger.get_logs_as_text()
    return "PROCESS_STOPPED_BY_USER" in logs


def run_verification_in_thread(file_path, retries, column_name):
    """
    Wrapper function to run the asyncio domain verification in a thread.
    This thread's ONLY job is to do work and set a completion flag.
    """

    def thread_target():
        completion_file = Path("verification_complete.flag")
        try:
            print("[DEBUG] Background thread starting verification...")
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
                    logger=logger,
                )
            )
            st.session_state.reachable_df = reachable_df
            st.session_state.verification_error = None
            print(
                "[DEBUG] Background thread completed successfully. Setting completion flag..."
            )

            # If contact crawling is enabled, start it now
            if st.session_state.enable_contact_crawling and reachable_df is not None:
                print(
                    f"[DEBUG] Starting contact form crawling on {len(reachable_df)} reachable domains..."
                )
                run_contact_crawling_in_thread(reachable_df)
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
    print("[DEBUG] Started background thread for verification")


def run_contact_crawling_in_thread(reachable_df):
    """
    Run contact form crawling on reachable domains in a separate thread.
    """

    def crawl_thread_target():
        try:
            print("[DEBUG] Starting contact crawling thread...")
            st.session_state.contact_crawl_running = True

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Limit to first 10 domains to avoid overwhelming
            domains_to_crawl = reachable_df["website"].head(10).tolist()
            contact_results = []

            for i, domain in enumerate(domains_to_crawl):
                try:
                    # Ensure domain has protocol
                    if not domain.startswith(("http://", "https://")):
                        domain = f"https://{domain}"

                    print(
                        f"[DEBUG] Crawling contact forms for {domain} ({i+1}/{len(domains_to_crawl)})"
                    )

                    crawler = ContactFormCrawler(
                        domain=domain,
                        max_pages=5,  # Limit pages for faster processing
                        max_depth=2,
                        crawl_strategy="priority_first",
                        rate_limit=1.0,  # Be polite
                        timeout=15,
                    )

                    forms = loop.run_until_complete(crawler.find_contact_forms())

                    if forms:
                        best_form = crawler.get_best_form()
                        contact_results.append(
                            {
                                "domain": domain,
                                "forms_found": len(forms),
                                "best_form_url": best_form["page_url"],
                                "confidence_score": best_form["confidence_score"],
                                "field_types": list(best_form["field_types"]),
                                "extracted_emails": best_form["extracted_contacts"][
                                    "emails"
                                ],
                                "extracted_phones": best_form["extracted_contacts"][
                                    "phones"
                                ],
                            }
                        )
                        print(f"[DEBUG] Found {len(forms)} forms for {domain}")
                    else:
                        print(f"[DEBUG] No forms found for {domain}")

                except Exception as e:
                    print(f"[DEBUG] Error crawling {domain}: {e}")
                    continue

            st.session_state.contact_crawl_results = contact_results
            print(
                f"[DEBUG] Contact crawling completed. Found forms on {len(contact_results)} domains"
            )

        except Exception as e:
            print(f"[DEBUG] Error in contact crawling thread: {e}")
        finally:
            st.session_state.contact_crawl_running = False

    t = threading.Thread(target=crawl_thread_target, daemon=True)
    t.start()
    print("[DEBUG] Started contact crawling thread")


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

    # Display logs from centralized logger
    logs = logger.get_logs_as_text(last_n=50)  # Show last 50 entries
    if logs:
        log_container = st.container(height=300)
        # Reverse order to show newest first
        log_lines = logs.split("\n")
        log_container.code("\n".join(reversed(log_lines)), language="log")


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
            print("[DEBUG] Fragment detected completion flag. Updating UI state...")
            st.session_state.process_running = False
            st.session_state.process_finished = True
            st.session_state.verification_complete_flag = False
            st.session_state.was_stopped_flag = check_if_stopped()
            # Clean up the completion file
            if completion_file.exists():
                completion_file.unlink()
                print("[DEBUG] Cleaned up completion file")
            print(
                "[DEBUG] UI state updated. process_running=False, process_finished=True"
            )
            print("[DEBUG] Triggering full app rerun to update Start button...")
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
    print("[DEBUG] Main logic detected completion flag. Updating UI state...")
    st.session_state.process_running = False
    st.session_state.process_finished = True
    st.session_state.verification_complete_flag = False
    st.session_state.was_stopped_flag = check_if_stopped()
    # Clean up the completion file
    if completion_file.exists():
        completion_file.unlink()
        print("[DEBUG] Main logic cleaned up completion file")
    print(
        "[DEBUG] Main logic updated UI state. process_running=False, process_finished=True"
    )
    print("[DEBUG] Main logic triggering full app rerun...")
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
        print("[DEBUG] Start button clicked. Initializing verification process...")
        st.session_state.process_running = True
        st.session_state.process_finished = False
        st.session_state.verification_complete_flag = False
        st.session_state.was_stopped_flag = False
        st.session_state.reachable_df = None
        logger.clear()  # Clear previous logs
        run_verification_in_thread(upload_path, num_times, "website")
        print("[DEBUG] Verification process initialized. process_running=True")
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

    # Display contact crawling status and results
    if st.session_state.enable_contact_crawling:
        st.markdown("### Contact Form Discovery")

        if st.session_state.contact_crawl_running:
            st.info("🔍 Searching for contact forms on reachable domains...")

        elif st.session_state.contact_crawl_results:
            results = st.session_state.contact_crawl_results
            st.success(f"✅ Contact forms found on {len(results)} domains!")

            # Create a nice display of results
            for result in results:
                with st.expander(
                    f"📧 {result['domain']} - {result['forms_found']} form(s) found"
                ):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.write(f"**Best Form URL:** {result['best_form_url']}")
                        st.write(
                            f"**Confidence Score:** {result['confidence_score']:.2f}"
                        )
                        st.write(f"**Field Types:** {', '.join(result['field_types'])}")

                    with col2:
                        if result["extracted_emails"]:
                            st.write(
                                f"**Emails Found:** {', '.join(result['extracted_emails'])}"
                            )
                        if result["extracted_phones"]:
                            st.write(
                                f"**Phones Found:** {', '.join(result['extracted_phones'])}"
                            )

            # Export contact results to CSV
            if results:
                contact_df = pd.DataFrame(results)
                contact_csv = contact_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download Contact Forms Data",
                    data=contact_csv,
                    file_name=f"{base_name}_contact_forms.csv",
                    use_container_width=True,
                )

        elif (
            st.session_state.process_finished
            and not st.session_state.contact_crawl_running
        ):
            st.info("No contact forms found on the reachable domains.")
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
