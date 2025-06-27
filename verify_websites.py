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

import argparse
import asyncio
import json
from pathlib import Path
import random

import pandas as pd
import numpy as np
import httpx
from tqdm.asyncio import tqdm as asyncio_tqdm


# Load user agents with fallback if file is missing
try:
    USER_AGENTS = json.load(open("src/form_autobot/user_agents.json"))
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
    input_path, retries=2, concurrency=100, column="website", log_path="process.log"
):
    """
    Verify domain reachability using a two-pass HEAD/GET method.

    Args:
        input_path: Path to the CSV file containing domains
        retries: Number of GET request retries for failed domains
        concurrency: Number of concurrent requests
        column: Name of the column containing domains
        log_path: Path to write logs to

    Returns:
        tuple: (reachable_df, report_df) DataFrames with results
    """
    # Reset log file
    with open(log_path, "w") as f:
        f.write(f"Starting verification process for {input_path}\n")

    # Log progress to file and update Streamlit UI
    def log_progress(pass_name, current, total, stage):
        percent = int((current / total) * 100) if total > 0 else 0
        progress_line = f"PROGRESS|pass={pass_name}|current={current}|total={total}|percent={percent}|stage={stage}\n"
        with open(log_path, "a") as f:
            f.write(progress_line)
            f.flush()

    # Ensure output directories exist
    Path("reachable").mkdir(parents=True, exist_ok=True)
    Path("all").mkdir(parents=True, exist_ok=True)

    # Generate output filenames
    base_name = Path(input_path).stem
    output_reachable_path = Path("reachable") / f"{base_name}_reachable.csv"
    output_report_path = Path("all") / f"{base_name}_report_all.csv"

    try:
        df = pd.read_csv(input_path)
        log_progress("READ INPUT", 0, len(df), "READ INPUT")
    except Exception as e:
        with open(log_path, "a") as f:
            f.write(f"Error reading CSV file: {e}\n")
            f.flush()
        return None, None

    if column not in df.columns:
        error_msg = f"Column '{column}' not found in the CSV file. Available columns: {list(df.columns)}"
        with open(log_path, "a") as f:
            f.write(error_msg + "\n")
            f.flush()
        return None, None

    urls_to_check = df[column].dropna().unique().tolist()
    final_results_map = {}  # Initialize as empty dict

    # Get random user agent
    headers = {"User-Agent": random.choice(USER_AGENTS)}

    with open(log_path, "a") as f:
        f.write(f"Found {len(urls_to_check)} unique domains to check\n")

    async with httpx.AsyncClient(headers=headers) as session:
        # --- Pass 1: High-Speed HEAD Scan ---
        log_progress("HEAD", 0, len(urls_to_check), "HEAD scan")

        semaphore = asyncio.Semaphore(concurrency)
        tasks = [is_reachable(url, session, semaphore, "HEAD") for url in urls_to_check]

        completed = 0
        for task in asyncio.as_completed(tasks):
            try:
                url, reachable, reason = await task
                final_results_map[url] = {
                    "url": url,
                    "reachable": reachable,
                    "remarks": reason,
                }
                completed += 1

                # Log progress periodically or at completion
                if completed % max(
                    1, min(10, len(urls_to_check) // 20)
                ) == 0 or completed == len(urls_to_check):
                    log_progress("HEAD", completed, len(urls_to_check), "HEAD scan")

            except Exception as e:
                pass  # Silently handle exceptions to keep progress going

        failed_urls = [
            url for url, result in final_results_map.items() if not result["reachable"]
        ]

        # --- Pass 2: Robust GET Scan on Failures ---
        if not failed_urls:
            with open(log_path, "a") as f:
                f.write(
                    "\nAll websites were reachable on the first pass. No second pass needed.\n"
                )
                f.flush()
        else:
            get_concurrency = max(10, concurrency // 2)
            with open(log_path, "a") as f:
                f.write(f"\nVerifying... (Pass 2: GET) - {len(failed_urls)} domains\n")
                f.flush()

            get_semaphore = asyncio.Semaphore(get_concurrency)
            failed_remarks_history = {url: [] for url in failed_urls}

            for i in range(retries):
                run_num = i + 1
                if not failed_urls:
                    with open(log_path, "a") as f:
                        f.write(
                            f"\nNo more URLs to check on run {run_num}. Stopping early.\n"
                        )
                        f.flush()
                    break

                tasks = [
                    is_reachable(url, session, get_semaphore, "GET")
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

                        completed += 1
                        if (
                            completed % max(1, min(5, total // 10)) == 0
                            or completed == total
                        ):
                            log_progress(
                                "GET",
                                completed,
                                total,
                                f"GET scan (run {run_num}/{retries})",
                            )
                    except Exception:
                        pass

                failed_urls = still_failing

            for url in failed_urls:
                if url in final_results_map and isinstance(
                    final_results_map[url], dict
                ):
                    final_results_map[url]["remarks"] = " -> ".join(
                        failed_remarks_history[url]
                    )

    # --- Generate Final Output ---
    report_df = pd.DataFrame(final_results_map.values())
    reachable_urls = report_df[report_df["reachable"]]["url"]
    reachable_df = df[df[column].isin(reachable_urls)]

    total_checked = len(urls_to_check)
    total_reachable = len(reachable_urls.unique())

    with open(log_path, "a") as f:
        f.write(f"\n--- Verification Complete ---\n")
        f.write(f"Total unique websites checked: {total_checked}\n")
        f.write(f"Reachable websites: {total_reachable}\n")
        f.write(f"Unreachable websites: {total_checked - total_reachable}\n")
        f.flush()

    # Save the files
    reachable_df.to_csv(output_reachable_path, index=False)
    report_df.to_csv(output_report_path, index=False)

    with open(log_path, "a") as f:
        f.write(
            f"Successfully saved {len(reachable_df)} rows to '{output_reachable_path}'\n"
        )
        f.write(
            f"Successfully saved full report for {len(report_df)} unique URLs to '{output_report_path}'\n"
        )
        f.flush()

    return reachable_df, report_df
