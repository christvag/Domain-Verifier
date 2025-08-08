import asyncio
import contextlib
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from urllib.parse import urljoin, urlparse

import pandas as pd
from bs4 import BeautifulSoup

from centralized_logger import get_logger
from database_manager import save_process_run_to_db

# Playwright
from playwright.async_api import async_playwright

BLOCK_RESOURCE_TYPES = {
    "image", "media", "font", "stylesheet", "xhr", "fetch", "websocket"
}
BLOCK_URL_SUBSTRINGS = [
    "googletagmanager", "google-analytics", "doubleclick", "facebook",
    "ads", "adservice", "hotjar", "segment", "optimizely", "mixpanel",
    "youtube", "vimeo", "tiktok", "fonts.googleapis", "gravatar"
]

CONTACT_KEYWORDS = [
    "contact", "support", "help", "about", "quote", "inquiry", "enquiry",
    "appointment", "booking", "message", "email", "call", "customer service"
]

@dataclass
class DomainMetrics:
    domain: str
    started_at: float
    finished_at: float = 0.0
    homepage_ms: int = 0
    candidate_links: int = 0
    visited_candidates: int = 0
    forms_found: int = 0
    timeouts: int = 0
    nav_errors: int = 0
    js_errors: int = 0
    blocked_requests: int = 0
    total_requests: int = 0
    bytes_saved_est: int = 0
    error: Optional[str] = None


@dataclass
class CrawlMetrics:
    run_started_at: float
    run_finished_at: float = 0.0
    domains_total: int = 0
    domains_processed: int = 0
    forms_total: int = 0
    timeouts: int = 0
    nav_errors: int = 0
    js_errors: int = 0
    blocked_requests: int = 0
    total_requests: int = 0
    bytes_saved_est: int = 0
    per_domain: List[DomainMetrics] = field(default_factory=list)

    def to_flat(self) -> Dict:
        d = asdict(self)
        d.pop("per_domain", None)
        return d


class BrowserPool:
    def __init__(self, headless: bool = True, contexts: int = 4, pages_per_context: int = 4, default_timeout_ms: int = 12000):
        self.headless = headless
        self.contexts_count = contexts
        self.pages_per_context = pages_per_context
        self.default_timeout_ms = default_timeout_ms
        self.playwright = None
        self.browser = None
        self.contexts = []
        self.context_queue: asyncio.Queue = asyncio.Queue()
        self.logger = get_logger()

    async def __aenter__(self):
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            headless=self.headless,
            args=[
                "--disable-gpu",
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-setuid-sandbox",
                "--disable-extensions",
                "--disable-background-networking",
                "--disable-background-timer-throttling",
                "--disable-sync",
                "--disable-default-apps",
                "--disable-features=IsolateOrigins,site-per-process,Translate,MediaRouter",
            ],
        )
        for _ in range(self.contexts_count):
            ctx = await self.browser.new_context(ignore_https_errors=True, java_script_enabled=True)
            ctx.set_default_timeout(self.default_timeout_ms)
            await self._install_blocking(ctx)
            self.contexts.append(ctx)
            # Fill queue per context pages
            for _ in range(self.pages_per_context):
                await self.context_queue.put(ctx)
        return self

    async def _install_blocking(self, context):
        async def route_handler(route):
            req = route.request
            url = req.url.lower()
            rtype = req.resource_type
            if rtype in BLOCK_RESOURCE_TYPES or any(s in url for s in BLOCK_URL_SUBSTRINGS):
                return await route.abort()
            return await route.continue_()
        await context.route("**/*", route_handler)

    async def acquire_context(self):
        return await self.context_queue.get()

    async def release_context(self, ctx):
        await self.context_queue.put(ctx)

    async def __aexit__(self, exc_type, exc, tb):
        for ctx in self.contexts:
            with contextlib.suppress(Exception):
                await ctx.close()
        if self.browser:
            with contextlib.suppress(Exception):
                await self.browser.close()
        if self.playwright:
            with contextlib.suppress(Exception):
                await self.playwright.stop()


class FastContactCrawler:
    def __init__(self, output_dir: str = "tmp", k_links: int = 3, per_domain_budget_s: int = 30):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.k_links = k_links
        self.per_domain_budget_s = per_domain_budget_s
        self.logger = get_logger()
        self.metrics = CrawlMetrics(run_started_at=time.time())

    async def crawl_csv(self, csv_file_path: str, website_col: str = "website", concurrency: int = 100,
                        headless: bool = True, contexts: int = 6, pages_per_context: int = 4) -> Tuple[str, Dict]:
        df = pd.read_csv(csv_file_path)
        if website_col not in df.columns:
            raise ValueError(f"CSV must contain column '{website_col}'")
        domains = [self._normalize_domain(d) for d in df[website_col].dropna().tolist()]
        self.metrics.domains_total = len(domains)

        results_path = self.output_dir / f"contact_urls_{int(time.time())}.csv"
        detail_path = self.output_dir / f"contact_urls_detail_{int(time.time())}.csv"
        results_file = results_path.open("w", encoding="utf-8")
        results_file.write("domain,contact_url,confidence\n")
        detail_records = []

        sem = asyncio.Semaphore(concurrency)

        async with BrowserPool(headless=headless, contexts=contexts, pages_per_context=pages_per_context) as pool:
            async def worker(domain: str):
                async with sem:
                    recs, dmetrics = await self._process_domain(pool, domain)
                    self.metrics.per_domain.append(dmetrics)
                    self.metrics.domains_processed += 1
                    self.metrics.forms_total += len(recs)
                    # stream to csv (reduces memory)
                    for r in recs:
                        results_file.write(f"{domain},{r['contact_url']},{r['confidence']:.3f}\n")
                        detail_records.append(r)

            tasks = [asyncio.create_task(worker(d)) for d in domains]
            for i, t in enumerate(asyncio.as_completed(tasks), 1):
                try:
                    await t
                except Exception as e:
                    self.logger.error(f"Domain task error: {e}")

        results_file.close()
        # Optional: write detail CSV
        if detail_records:
            pd.DataFrame(detail_records).to_csv(detail_path, index=False)

        self.metrics.run_finished_at = time.time()

        return str(results_path), self.metrics.to_flat()

    async def _process_domain(self, pool: BrowserPool, domain: str) -> Tuple[List[Dict], DomainMetrics]:
        started = time.time()
        dmetrics = DomainMetrics(domain=domain, started_at=started)
        ctx = await pool.acquire_context()
        try:
            page = await ctx.new_page()
            req_counts = {"total": 0, "blocked": 0}
            bytes_saved_est = 0

            page.on("request", lambda r: self._on_req(r, req_counts))
            page.on("requestfinished", lambda r: self._on_req_finished(r))
            page.on("console", lambda m: self._on_console(m, dmetrics))

            # 1) Home page
            try:
                await page.goto(domain, wait_until="domcontentloaded")
            except Exception as e:
                dmetrics.nav_errors += 1
                dmetrics.error = f"homepage: {type(e).__name__}"
                return [], self._finish_metrics(dmetrics, req_counts, bytes_saved_est)

            dmetrics.homepage_ms = int((time.time() - started) * 1000)

            # 2) Extract links quickly via JS (no BeautifulSoup needed)
            anchors = await page.evaluate("""
                () => Array.from(document.querySelectorAll('a[href]')).map(a => ({
                    text: (a.innerText || '').trim(),
                    href: a.getAttribute('href')
                }))
            """)
            base = self._origin(domain)
            same_domain = []
            for a in anchors:
                href = a.get("href") or ""
                text = (a.get("text") or "").strip()
                if not href or href.startswith(("javascript:", "mailto:", "tel:", "#")):
                    continue
                full = href if href.startswith("http") else urljoin(base, href)
                if urlparse(full).netloc != urlparse(base).netloc:
                    continue
                same_domain.append({"text": text, "full": full})

            # 3) Pre-filter contact links
            candidates = self._rank_candidates(same_domain)
            dmetrics.candidate_links = len(candidates)
            candidates = candidates[: self.k_links]

            # 4) Visit top-k candidates within per-domain budget
            deadline = started + self.per_domain_budget_s
            found: List[Dict] = []
            for c in candidates:
                if time.time() > deadline:
                    dmetrics.timeouts += 1
                    break
                try:
                    dmetrics.visited_candidates += 1
                    await page.goto(c["full"], wait_until="domcontentloaded")
                    has_form, info = await self._detect_form(page)
                    if has_form:
                        found.append({
                            "domain": domain,
                            "contact_url": c["full"],
                            "link_text": c["text"],
                            "confidence": info.get("confidence", 0.7),
                            "form_type": info.get("form_type", "html_form"),
                        })
                except Exception:
                    dmetrics.nav_errors += 1
                    continue

            return found, self._finish_metrics(dmetrics, req_counts, bytes_saved_est)
        finally:
            with contextlib.suppress(Exception):
                await page.close()
            await pool.release_context(ctx)

    def _rank_candidates(self, links: List[Dict]) -> List[Dict]:
        ranked = []
        for l in links:
            text = f"{l['text']}".lower()
            url = l["full"].lower()
            score = 0
            for k in CONTACT_KEYWORDS:
                if k in text:
                    score += 2
                if k in url:
                    score += 1
            # small boost for shorter contact URLs
            score -= min(len(url), 120) / 120.0
            ranked.append((score, l))
        ranked.sort(key=lambda x: x[0], reverse=True)
        return [r[1] for r in ranked]

    async def _detect_form(self, page) -> Tuple[bool, Dict]:
        # Fast DOM-side checks (no soup)
        has_form = await page.evaluate("""
            () => {
                const forms = Array.from(document.querySelectorAll('form'));
                const score = forms.map(f => {
                    const inputs = Array.from(f.querySelectorAll('input, textarea, select'));
                    let c = 0;
                    for (const el of inputs) {
                        const type = (el.getAttribute('type') || '').toLowerCase();
                        const name = (el.getAttribute('name') || '').toLowerCase();
                        const ph = (el.getAttribute('placeholder') || '').toLowerCase();
                        if (type === 'email' || name.includes('email') || ph.includes('email')) c++;
                        if (name.includes('name') || ph.includes('name')) c++;
                        if (name.includes('phone') || ph.includes('phone') || type === 'tel') c++;
                        if (el.tagName.toLowerCase() === 'textarea' || name.includes('message') || ph.includes('message')) c++;
                    }
                    return {inputs: inputs.length, contactFields: c};
                });
                const maxC = score.reduce((m, s) => Math.max(m, s.contactFields), 0);
                const maxInputs = score.reduce((m, s) => Math.max(m, s.inputs), 0);
                const isContact = maxC >= 2 || (maxInputs >= 3 && maxC >= 1);
                return {isContact, conf: Math.min(1, 0.5 + maxC * 0.15)};
            }
        """)
        if has_form and has_form.get("isContact"):
            return True, {"confidence": has_form.get("conf", 0.7), "form_type": "html_form"}

        # Iframe services check
        iframe_hit = await page.evaluate("""
            () => {
                const ifr = Array.from(document.querySelectorAll('iframe[src]'));
                const services = ['typeform.com','forms.gle','google.com/forms','jotform.com','wufoo.com',
                                  'formstack.com','hubspot.com','mailchimp.com','formspree.io','tally.so'];
                for (const f of ifr) {
                    const src = f.getAttribute('src').toLowerCase();
                    const ok = services.some(s => src.includes(s)) ||
                               src.includes('contact') || src.includes('form');
                    if (ok) return true;
                }
                return false;
            }
        """)
        if iframe_hit:
            return True, {"confidence": 0.7, "form_type": "iframe_form"}

        return False, {}

    def _on_req(self, request, counters):
        counters["total"] += 1

    def _on_req_finished(self, request):
        pass

    def _on_console(self, msg, dmetrics: DomainMetrics):
        if msg.type in ("error", "warning"):
            dmetrics.js_errors += 1

    def _finish_metrics(self, dmetrics: DomainMetrics, req_counts, bytes_saved_est: int) -> DomainMetrics:
        dmetrics.finished_at = time.time()
        dmetrics.total_requests = req_counts["total"]
        dmetrics.blocked_requests = req_counts["blocked"]
        dmetrics.bytes_saved_est = bytes_saved_est
        # roll up
        self.metrics.blocked_requests += dmetrics.blocked_requests
        self.metrics.total_requests += dmetrics.total_requests
        self.metrics.timeouts += dmetrics.timeouts
        self.metrics.nav_errors += dmetrics.nav_errors
        self.metrics.js_errors += dmetrics.js_errors
        self.metrics.bytes_saved_est += dmetrics.bytes_saved_est
        return dmetrics

    def _normalize_domain(self, d: str) -> str:
        if not isinstance(d, str) or not d.strip():
            return ""
        d = d.strip()
        if not d.startswith(("http://", "https://")):
            d = "https://" + d
        return d

    def _origin(self, url: str) -> str:
        u = urlparse(url)
        return f"{u.scheme}://{u.netloc}"


async def crawl_from_csv(csv_file_path: str, website_col: str = "website",
                         output_dir: str = "tmp", concurrency: int = 100,
                         headless: bool = True) -> Tuple[str, Dict]:
    crawler = FastContactCrawler(output_dir=output_dir)
    return await crawler.crawl_csv(
        csv_file_path=csv_file_path,
        website_col=website_col,
        concurrency=concurrency,
        headless=headless,
        contexts=6,
        pages_per_context=4,
    )