"""
Website Reachability Checker (Two-Pass)

This script uses an advanced two-pass system to accurately and efficiently
check the reachability of websites from a CSV file.

**Workflow:**
1.  **Pass 1 (High-Speed HEAD Scan):** It first scans all unique URLs using
    the fast `HEAD` method with high concurrency. This quickly finds all the
    easily reachable sites.
2.  **Pass 2 (Robust GET Scan):** For the URLs that failed the first pass,
    it performs a more intensive scan using the more compatible `GET` method.
    This pass uses retries to handle intermittent network errors and ensure
    the highest possible accuracy.

This hybrid approach provides the speed of HEAD requests with the accuracy of GET
requests, giving you the most reliable results in the shortest amount of time.

It generates two output files:
1. A CSV containing the original data for only the reachable websites.
2. A comprehensive report CSV detailing the final status of every unique website checked.

-------------------------------------------------------------------------------
Setup and Usage
-------------------------------------------------------------------------------

**1. Python Installation:**
   Ensure you have Python 3 installed. If you don't, you can download it from
   https://www.python.org/downloads/.

   Throughout these instructions, `python` and `pip` are used. If your system
   has these aliased to `python3` and `pip3`, these commands will work as is.
   If not, you may need to explicitly use `python3` and `pip3`.

**2. Dependency Installation:**
   This script requires a few Python packages. Install them using pip:

   pip install pandas httpx tqdm

**3. Running the Script:**
   Execute the script from your terminal, providing the path to your input CSV.

   Basic usage (3 retries, 100 concurrency):
   python verify_websites.py <input_file.csv>

   Specifying the number of retries:
   python verify_websites.py <input_file.csv> --retries 5

   Specifying a different concurrency limit:
   python verify_websites.py <input_file.csv> --concurrency 50

"""

import asyncio
import json
from pathlib import Path
import random
import time
import math

import pandas as pd
import httpx

from database_manager import save_run_to_db
from centralized_logger import get_logger


# Load user agents with fallback if file is missing
try:
    USER_AGENTS = json.load(open("user_agents.json"))
except (FileNotFoundError, OSError, json.JSONDecodeError):
    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.1 Safari/605.1.15",
    ]
headers = {
    "User-Agent": random.choice(USER_AGENTS),
}


async def is_reachable(
    url: str,
    session: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    method: str = "HEAD",
) -> tuple[str, bool, str]:
    """
    Checks if a URL is reachable, respecting a semaphore.
    Uses a specified HTTP method (HEAD or GET) for the check.
    """
    original_url = url
    if not isinstance(url, str) or not url.strip():
        return original_url, False, "Invalid URL (empty)"

    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    async with semaphore:
        try:
            if method == "GET":
                async with session.stream(
                    "GET", url, timeout=10, follow_redirects=True
                ) as response:
                    status_code = response.status_code
                    async for _ in response.aiter_bytes(chunk_size=512):
                        break
            else:  # Default to HEAD
                response = await session.head(url, timeout=10, follow_redirects=True)
                status_code = response.status_code

            if status_code < 400:
                return original_url, True, f"Success ({status_code})"
            elif 400 <= status_code < 500:
                return original_url, False, f"Client Error ({status_code})"
            else:
                return original_url, False, f"Server Error ({status_code})"
        except httpx.ConnectTimeout:
            return original_url, False, "Connection Timeout"
        except httpx.ReadTimeout:
            return original_url, False, "Read Timeout"
        except httpx.WriteTimeout:
            return original_url, False, "Write Timeout"
        except httpx.PoolTimeout:
            return original_url, False, "Pool Timeout"
        except httpx.ConnectError:
            return original_url, False, "Connection Error"
        except httpx.TooManyRedirects:
            return original_url, False, "Too Many Redirects"
        except httpx.UnsupportedProtocol as e:
            return original_url, False, f"Unsupported Protocol ({e.request.url.scheme})"
        except httpx.ProxyError:
            return original_url, False, "Proxy Error"
        except httpx.RequestError as e:
            return original_url, False, f"Request Error: {type(e).__name__}"
        except Exception as e:
            return original_url, False, f"Unexpected Error: {str(e)}"


async def verify_domains(
    input_path,
    start_time,
    retries=2,
    concurrency=100,
    column="website",
    logger=None,
    chunksize=15000,
):
    """
    Verify domain reachability using a two-pass HEAD/GET method with batching.

    Args:
        input_path: Path to the CSV file containing domains.
        start_time: The timestamp when the process began.
        retries: Number of GET request retries for failed domains.
        concurrency: Number of concurrent requests.
        column: Name of the column containing domains.
        logger: Centralized logger instance.
        chunksize: Number of rows to process in each batch.

    Returns:
        tuple: (final_reachable_df, final_report_df) DataFrames with results.
    """
    # Use provided logger or get default
    if logger is None:
        logger = get_logger()

    stop_signal_file = Path("stop_processing.flag")
    if stop_signal_file.exists():
        stop_signal_file.unlink()  # Clean up from previous runs

    logger.info(f"Starting verification process for {input_path}")

    def log_progress(pass_name, current, total, stage):
        percent = int((current / total) * 100) if total > 0 else 0
        logger.progress(f"{stage} ({current}/{total})", percent)

    def log_info(message):
        """Helper function to log info messages"""
        logger.info(message)

    def log_warning(message):
        """Helper function to log warning messages"""
        logger.warning(message)

    Path("reachable").mkdir(parents=True, exist_ok=True)
    Path("all").mkdir(parents=True, exist_ok=True)

    base_name = Path(input_path).stem
    output_reachable_path = Path("reachable") / f"{base_name}_reachable.csv"
    output_report_path = Path("all") / f"{base_name}_report_all.csv"

    # --- Calculate total batches ---
    total_batches = "N/A"
    try:
        # Count lines for progress estimation without loading the whole file
        with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
            total_rows = sum(1 for _ in f) - 1  # -1 for header
        if total_rows > 0:
            total_batches = math.ceil(total_rows / chunksize)
        else:
            total_batches = 1  # Even an empty file is one "batch" to process
            total_rows = 0  # ensure it's not negative

        log_info(
            f"Input file has {total_rows} data rows. Processing in {total_batches} batches of up to {chunksize}."
        )
    except Exception as e:
        log_warning(
            f"Could not pre-calculate total batches. Will proceed without batch count. Error: {e}"
        )

    # --- State Management: Load already processed URLs to allow for resuming ---
    processed_urls = set()
    if output_report_path.exists():
        try:
            report_df_existing = pd.read_csv(output_report_path)
            if "url" in report_df_existing.columns:
                processed_urls = set(report_df_existing["url"].dropna().unique())
            log_info(
                f"Resuming session. Found {len(processed_urls)} already processed URLs."
            )
        except (pd.errors.EmptyDataError, FileNotFoundError):
            log_info("Report file exists but is empty or invalid. Starting fresh.")

    try:
        df_iterator = pd.read_csv(input_path, chunksize=chunksize, on_bad_lines="warn")
    except Exception as e:
        logger.error(f"Fatal error reading CSV file: {e}")
        return None, None

    # --- Batch Processing ---
    batch_num = 0
    stop_requested = False
    async with httpx.AsyncClient(headers=headers) as session:
        for df_batch in df_iterator:
            batch_num += 1

            # Check for stop signal at the beginning of each batch
            if stop_signal_file.exists():
                log_info("Stop signal detected. Halting processing.")
                stop_requested = True

            if stop_requested:
                break

            if column not in df_batch.columns:
                error_msg = f"Column '{column}' not found in the CSV file. Available columns: {list(df_batch.columns)}"
                logger.error(error_msg)
                continue  # Skip to the next batch

            urls_in_batch = df_batch[column].dropna().unique()
            urls_to_check = [url for url in urls_in_batch if url not in processed_urls]

            if not urls_to_check:
                log_info(
                    f"Batch {batch_num}: All {len(urls_in_batch)} URLs already processed. Skipping."
                )
                continue

            log_info(
                f"--- Processing Batch {batch_num}/{total_batches}: {len(urls_to_check)} new URLs ---"
            )

            final_results_map = {
                url: {"url": url, "reachable": False, "remarks": "Not Processed"}
                for url in urls_to_check
            }

            # --- Pass 1: High-Speed HEAD Scan ---
            log_progress(
                "HEAD", 0, len(urls_to_check), f"Batch {batch_num} - HEAD scan"
            )
            semaphore = asyncio.Semaphore(concurrency)
            tasks = [
                asyncio.create_task(is_reachable(url, session, semaphore, "HEAD"))
                for url in urls_to_check
            ]

            completed = 0
            for task in asyncio.as_completed(tasks):
                try:
                    url, reachable, reason = await task
                    final_results_map[url] = {
                        "url": url,
                        "reachable": reachable,
                        "remarks": reason,
                    }
                except asyncio.CancelledError:
                    # This is an expected error when we cancel tasks, so we can safely ignore it.
                    pass
                completed += 1
                if completed % 50 == 0 and stop_signal_file.exists():
                    stop_requested = True
                    break
                if completed % max(
                    1, len(urls_to_check) // 100
                ) == 0 or completed == len(urls_to_check):
                    log_progress(
                        "HEAD",
                        completed,
                        len(urls_to_check),
                        f"Batch {batch_num} - HEAD scan",
                    )

            if stop_requested:
                log_info(
                    "Stop signal detected during HEAD scan. Cancelling remaining tasks for this batch..."
                )
                print("[DEBUG] Stop signal detected during HEAD; cancelling tasks...")
                stop_requested = True
                # Cancel all outstanding tasks to ensure a clean shutdown
                for task in tasks:
                    task.cancel()
                # Wait for all tasks to acknowledge the cancellation
                await asyncio.gather(*tasks, return_exceptions=True)

            failed_urls = [
                url
                for url, result in final_results_map.items()
                if not result["reachable"]
            ]

            # --- Pass 2: Robust GET Scan on Failures ---
            if failed_urls and not stop_requested:
                get_concurrency = max(10, concurrency // 2)
                get_semaphore = asyncio.Semaphore(get_concurrency)
                failed_remarks_history = {url: [] for url in failed_urls}

                for i in range(retries):
                    run_num = i + 1
                    if not failed_urls:
                        break  # All failures resolved in previous retries

                    if stop_requested:
                        break

                    tasks = [
                        asyncio.create_task(
                            is_reachable(url, session, get_semaphore, "GET")
                        )
                        for url in failed_urls
                    ]
                    still_failing = []

                    completed = 0
                    total = len(failed_urls)
                    for task in asyncio.as_completed(tasks):
                        try:
                            url, reachable, reason = await task
                            if reachable:
                                final_results_map[url] = {
                                    "url": url,
                                    "reachable": True,
                                    "remarks": reason,
                                }
                            else:
                                still_failing.append(url)
                                failed_remarks_history[url].append(reason)
                        except asyncio.CancelledError:
                            # This is an expected error when we cancel tasks, so we can safely ignore it.
                            pass

                        completed += 1
                        if completed % 20 == 0 and stop_signal_file.exists():
                            stop_requested = True
                            break
                        if completed % max(1, total // 100) == 0 or completed == total:
                            log_progress(
                                "GET",
                                completed,
                                total,
                                f"Batch {batch_num} - GET run {run_num}/{retries}",
                            )

                    if stop_requested:
                        # Cancel any remaining tasks from this GET pass
                        for task in tasks:
                            task.cancel()
                        await asyncio.gather(*tasks, return_exceptions=True)
                        log_info(
                            "Stop signal during GET pass. Cancelling remaining tasks..."
                        )
                        print(
                            "[DEBUG] Stop signal detected during GET; cancelling tasks..."
                        )
                        stop_requested = True

                    failed_urls = still_failing

                for url in failed_urls:
                    final_results_map[url]["remarks"] = " -> ".join(
                        failed_remarks_history[url]
                    )

            # --- Append results of the batch to output files ---
            report_df_batch = pd.DataFrame(final_results_map.values())
            if not report_df_batch.empty:
                report_df_batch.to_csv(
                    output_report_path,
                    mode="a",
                    header=not output_report_path.exists()
                    or output_report_path.stat().st_size == 0,
                    index=False,
                )

                reachable_urls_batch = report_df_batch[report_df_batch["reachable"]][
                    "url"
                ]
                reachable_df_batch = df_batch[
                    df_batch[column].isin(reachable_urls_batch)
                ]

                if not reachable_df_batch.empty:
                    reachable_df_batch.to_csv(
                        output_reachable_path,
                        mode="a",
                        header=not output_reachable_path.exists()
                        or output_reachable_path.stat().st_size == 0,
                        index=False,
                    )

            # Update processed_urls set for the next batch
            processed_urls.update(urls_to_check)

            if stop_requested:
                logger.info("PROCESS_STOPPED_BY_USER")
                print(
                    "[DEBUG] Verification process stopped gracefully after user request."
                )
                break

    # --- Finalization Step ---
    if stop_signal_file.exists():
        stop_signal_file.unlink()

    if not output_report_path.exists() or output_report_path.stat().st_size == 0:
        log_info("No URLs were processed. Exiting.")
        return pd.DataFrame(), pd.DataFrame()

    # Read the final, complete report to get aggregate statistics
    final_report_df = pd.read_csv(output_report_path)
    final_reachable_df = (
        pd.read_csv(output_reachable_path)
        if output_reachable_path.exists()
        else pd.DataFrame(columns=df_batch.columns)
    )

    total_checked = len(final_report_df["url"].unique())
    total_reachable = len(final_report_df[final_report_df["reachable"]]["url"].unique())

    # --- Save to Database ---
    end_time = time.time()
    processing_time = end_time - start_time
    save_run_to_db(
        filename=Path(input_path).name,
        processing_time=processing_time,
        reachable_count=total_reachable,
        unreachable_count=(total_checked - total_reachable),
        reachable_filepath=str(output_reachable_path),
        report_filepath=str(output_report_path),
    )

    log_info("--- Verification Complete ---")
    log_info(f"Total unique websites checked: {total_checked}")
    log_info(f"Reachable websites: {total_reachable}")
    log_info(f"Unreachable websites: {total_checked - total_reachable}")

    print("[DEBUG] verify_domains() exiting cleanly.")
    return final_reachable_df, final_report_df
