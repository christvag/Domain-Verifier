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


USER_AGENTS = json.load(open("src/form_autobot/user_agents.json"))
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


async def main():
    """
    Main function implementing the two-pass (HEAD then GET) check.
    """
    parser = argparse.ArgumentParser(
        description="Check reachability of websites using a two-pass HEAD/GET method.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("input_csv", help="Path to the input CSV file.")
    parser.add_argument(
        "--column", default="website", help="Name of the column containing website URLs."
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=100,
        help="Number of parallel requests for the initial HEAD scan. (Default: 100).",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Number of times to retry failed websites with GET requests. (Default: 2).",
    )
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    if not input_path.is_file():
        print(f"Error: Input file not found at {input_path}")
        return

    # Generate dynamic output filenames
    base_name = input_path.stem
    output_reachable_path = input_path.parent / f"{base_name}_reachable.csv"
    output_report_path = input_path.parent / f"{base_name}_report_all.csv"

    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    if args.column not in df.columns:
        print(
            f"Error: Column '{args.column}' not found in the CSV file. Available columns: {list(df.columns)}"
        )
        return

    urls_to_check = df[args.column].dropna().unique().tolist()
    final_results_map = {url: None for url in urls_to_check}

    async with httpx.AsyncClient(headers=headers) as session:
        # --- Pass 1: High-Speed HEAD Scan ---
        print(
            f"--- Pass 1: Starting high-speed HEAD scan for {len(urls_to_check)} URLs (Concurrency: {args.concurrency}) ---"
        )
        semaphore = asyncio.Semaphore(args.concurrency)
        tasks = [is_reachable(url, session, semaphore, "HEAD") for url in urls_to_check]
        for task in asyncio_tqdm.as_completed(tasks, total=len(tasks), desc="Pass 1 (HEAD)"):
            try:
                url, reachable, reason = await task
                final_results_map[url] = {"url": url, "reachable": reachable, "remarks": reason}
            except Exception as e:
                print(f"\nError processing a task result: {e}")

        failed_urls = [url for url, result in final_results_map.items() if not result["reachable"]]

        # --- Pass 2: Robust GET Scan on Failures ---
        if not failed_urls:
            print("\nAll websites were reachable on the first pass. No second pass needed.")
        else:
            get_concurrency = max(10, args.concurrency // 2)
            print(
                f"\n--- Pass 2: Starting robust GET scan for {len(failed_urls)} failed URLs ({args.retries} retries, Concurrency: {get_concurrency}) ---"
            )
            get_semaphore = asyncio.Semaphore(get_concurrency)

            # This will store the history of remarks for URLs that keep failing
            failed_remarks_history = {url: [] for url in failed_urls}

            for i in range(args.retries):
                run_num = i + 1
                if not failed_urls:
                    print(f"\nNo more URLs to check on run {run_num}. Stopping early.")
                    break

                tasks = [is_reachable(url, session, get_semaphore, "GET") for url in failed_urls]

                # List for URLs that fail THIS run
                still_failing = []

                for task in asyncio_tqdm.as_completed(
                    tasks, total=len(tasks), desc=f"Pass 2 (GET) Run {run_num}/{args.retries}"
                ):
                    try:
                        url, reachable, reason = await task
                        if reachable:
                            # Success! Update the main results map and we're done with this URL.
                            final_results_map[url] = {
                                "url": url,
                                "reachable": True,
                                "remarks": reason,
                            }
                        else:
                            # Failure. Add to the list for the next retry and record the error.
                            still_failing.append(url)
                            failed_remarks_history[url].append(reason)
                    except Exception as e:
                        print(f"\nError processing a task result: {e}")

                print(
                    f"URLs that turned to succcess from failure on this run: {len(failed_urls) - len(still_failing)}"
                )

                # The list for the next loop is the list of URLs that just failed.
                failed_urls = still_failing

            # After all retries, for any URL that never succeeded, consolidate its error remarks.
            for url in failed_urls:
                final_results_map[url]["remarks"] = " -> ".join(failed_remarks_history[url])

    # --- Generate Final Output ---
    report_df = pd.DataFrame(final_results_map.values())
    reachable_urls = report_df[report_df["reachable"]]["url"]
    reachable_df = df[df[args.column].isin(reachable_urls)]

    total_checked = len(urls_to_check)
    total_reachable = len(reachable_urls.unique())
    print(f"\n--- Verification Complete ---")
    print(f"Total unique websites checked: {total_checked}")
    print(f"Reachable websites: {total_reachable}")
    print(f"Unreachable websites: {total_checked - total_reachable}")
    print(f"-----------------------------")

    reachable_df.to_csv(output_reachable_path, index=False)
    print(f"Successfully saved {len(reachable_df)} rows to '{output_reachable_path}'")

    report_df.to_csv(output_report_path, index=False)
    print(
        f"Successfully saved full report for {len(report_df)} unique URLs to '{output_report_path}'"
    )


if __name__ == "__main__":
    asyncio.run(main())
