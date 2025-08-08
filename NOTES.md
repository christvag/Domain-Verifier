# Fast Contact Crawler (Playwright + asyncio) — Process Flow

Goal: A drop‑in, faster crawler that streams live results and collects metrics without breaking existing flows.

Returns: (output_csv_path, run_metrics_dict, metrics_json_path)

Legend:
- [Emit] → streams out to UI via callback
- [Write] → appends to disk incrementally
- [Metrics] → updates run counters

```
┌───────────────────────────────────────────────────────────────────────────────┐
│ Caller (domainVerifier or script)                                             │
│   crawl_from_csv(csv_file_path, website_col, output_dir,                      │
│                 concurrency, headless, on_contact, on_progress)               │
└───────────────┬───────────────────────────────────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│ Setup                                                                         │
│ - Create output_dir                                                           │
│ - Init logger + metrics (domains_total, processed, forms_total, timers, etc.) │
│ - Open results CSV for append: "domain,contact_url,confidence"                │
│ - Load CSV → normalize + filter domains                                       │
│ - metrics.domains_total = N                                                   │
└───────────────┬───────────────────────────────────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│ Playwright Pool                                                               │
│ - Launch Playwright (headless/headful)                                        │
│ - Warm up contexts/pages (contexts×pages_per_context)                         │
│ - Block heavy resources (image, media, font, stylesheet, xhr, fetch, ws)      │
│ - Reuse pages across tasks for speed                                          │
└───────────────┬───────────────────────────────────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│ Async Orchestrator                                                            │
│ - Create asyncio.Semaphore(concurrency)                                       │
│ - For each domain → schedule worker(domain)                                   │
│ - Consume tasks with asyncio.as_completed(...)                                │
└───────────────┬───────────────────────────────────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│ worker(domain)                                                                │
│ 1) Acquire semaphore                                                          │
│ 2) Get page from pool                                                         │
│ 3) Navigate to domain                                                         │
│ 4) Collect candidate contact links (heuristics/ML-ready structure)            │
│ 5) Visit top candidates (few pages)                                           │
│ 6) Detect forms/contact endpoints                                             │
│ 7) For each found contact URL:                                                │
│      • [Write] append "domain,contact_url,confidence" to results CSV          │
│      • [Emit] on_contact({...}) for live UI table updates                     │
│      • [Metrics] forms_total += 1                                             │
│ 8) Release page back to pool                                                  │
│ 9) [Metrics] domains_processed += 1                                           │
│10) [Emit] on_progress(processed,total,last_domain)                            │
└───────────────┬───────────────────────────────────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│ Streaming During Run                                                          │
│ - on_contact(row) → UI updates table incrementally (no wait for completion)   │
│ - on_progress(p,t,domain) → UI progress bar + label                           │
│ - logger.progress(stage, percent) → centralized log                           │
└───────────────┬───────────────────────────────────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│ Completion                                                                    │
│ - Close results CSV                                                           │
│ - Save detail CSV (optional)                                                  │
│ - Build run summary (dict)                                                    │
│ - [Write] metrics JSON (summary + per-domain if collected)                    │
│ - Return (results_csv_path, run_metrics_dict, metrics_json_path)              │
└───────────────────────────────────────────────────────────────────────────────┘
```

Why it’s faster:
- High concurrency with asyncio workers guarded by a semaphore.
- Playwright page/context reuse reduces cold-start time per domain.
- Resource blocking (images, fonts, media, xhr/ws) cuts network + render cost.
- Append-only CSV writing avoids keeping everything in memory.
- Streaming callbacks decouple UI from crawl speed and avoid main-thread stalls.

Non-breaking drop‑in design:
- Same single entry point: crawl_from_csv(...)
- Same return contract: paths + summary dict → existing DB/storage flows keep working.
- Optional callbacks (on_contact, on_progress) enable live UI without changing storage.