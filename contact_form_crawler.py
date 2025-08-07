#!/usr/bin/env python3
"""
Contact Form Crawler with CSV Support

- Reads domains from CSV file
- Crawls websites to find contact forms
- Outputs contact URLs to CSV file
"""

import os
import warnings
import pandas as pd
import asyncio
import time
from pathlib import Path
from typing import List, Dict, Optional
from urllib.parse import urljoin, urlparse

# Enable tokenizer parallelism but limit it to avoid forking issues
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["OMP_NUM_THREADS"] = "1"  # Limit OpenMP threads

# Suppress transformer deprecation warnings
warnings.filterwarnings(
    "ignore", message=".*encoder_attention_mask.*", category=FutureWarning
)

import json
import pickle
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from bs4 import BeautifulSoup
from centralized_logger import get_logger

try:
    from sentence_transformers import SentenceTransformer
    TRANSFORMER_AVAILABLE = True
except ImportError:
    TRANSFORMER_AVAILABLE = False


class ContactFormCrawler:
    """
    Contact form crawler with CSV support:
    - Can process individual domains (original functionality)
    - Can process multiple domains from CSV file
    - Outputs contact URLs to CSV
    """

    def __init__(self, domain: str = None, csv_file_path: str = None, output_dir: str = "tmp"):
        # Initialize for single domain or CSV mode
        self.domain = domain.rstrip("/") if domain else None
        self.csv_file_path = csv_file_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Logger setup
        self.logger = get_logger()
        
        # Results storage
        self.found_forms = []  # For single domain mode
        self.crawled_urls = set()
        self.results = []  # For CSV mode - will store all contact URLs
        
        # Initialize ML components
        self.sentence_model = None
        self.classifier = None
        self.model_dir = Path("models")
        self._models_loaded = False
        
        # Load ML models
        self._load_pretrained_models()

    def _load_pretrained_models(self):
        """Load pre-trained models from disk"""
        if self._models_loaded:
            return

        if not TRANSFORMER_AVAILABLE:
            self.logger.warning("⚠️ sentence-transformers not available, using heuristics")
            return

        try:
            # Check if models exist
            sentence_model_path = self.model_dir / "sentence_model"
            classifier_path = self.model_dir / "contact_classifier.pkl"

            if not sentence_model_path.exists() or not classifier_path.exists():
                self.logger.info(f"⚠️ Pre-trained models not found. Run contact_link_trainer.py first")
                self.logger.info(f"   Looking for: {sentence_model_path} and {classifier_path}")
                return

            self.logger.info(f"📥 Loading pre-trained models...")

            # Load sentence transformer
            self.sentence_model = SentenceTransformer(str(sentence_model_path))

            # Load classifier
            with open(classifier_path, "rb") as f:
                self.classifier = pickle.load(f)

            self.logger.info(f"✅ Pre-trained models loaded successfully")
            self._models_loaded = True

        except Exception as e:
            self.logger.info(f"⚠️ Failed to load pre-trained models: {e}")
            self.logger.info(f"   Run: python contact_link_trainer.py")
            self.sentence_model = None
            self.classifier = None

    async def crawl_csv_domains(
        self, 
        save_detail: bool = False, 
        column_name: str = "contact_url",
        on_result=None  # <-- Add this
    ) -> str:
        """
        Process all domains from CSV file and output results as CSV
        
        Args:
            save_detail: If True, also save a detailed version with all fields
            column_name: Name for the contact URL column in the CSV
            
        Returns:
            str: Path to output CSV file
        """
        if not self.csv_file_path or not Path(self.csv_file_path).exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_file_path}")
            
        # Read CSV file
        import pandas as pd
        df = pd.read_csv(self.csv_file_path)
        
        # Check for website column
        if 'website' not in df.columns:
            raise ValueError("CSV must have a 'website' column")
            
        domains = df['website'].dropna().tolist()
        self.logger.info(f"🚀 Starting contact form crawling for {len(domains)} domains")
        
        # Process domains with concurrency control
        semaphore = asyncio.Semaphore(5)  # Limit to 5 concurrent crawls
        
        for i, domain in enumerate(domains, 1):
            domain = domain.strip()
            if not domain:
                continue

            self.logger.info(f"[{i}/{len(domains)}] Processing: {domain}")

            try:
                async with semaphore:
                    contact_urls = await self._crawl_single_domain(domain)
                    for url in contact_urls:
                        self.results.append(url)
                        if on_result:
                            on_result(url)  # <-- Call the callback for each result
                    self.logger.info(f"✅ Found {len(contact_urls)} contact URLs for {domain}")
            except Exception as e:
                self.logger.error(f"Error processing {domain}: {str(e)}")

            self.logger.info(f"📊 Progress: {i}/{len(domains)} domains processed, {len(self.results)} total contact URLs found")
            
        # Save results to CSV with custom column name
        output_path = self._save_results_to_csv(save_detail=save_detail, column_name=column_name)
        
        self.logger.info(f"✅ Crawling completed. Found {len(self.results)} contact URLs across {len(domains)} domains")
        self.logger.info(f"📄 Results saved to: {output_path}")
        
        return output_path
        
    async def _crawl_single_domain_with_semaphore(self, domain: str, semaphore: asyncio.Semaphore) -> List[Dict]:
        """Helper to crawl a single domain with semaphore control"""
        async with semaphore:
            return await self._crawl_single_domain(domain)

    async def _crawl_single_domain(self, domain: str) -> List[Dict]:
        """
        Process a single domain to find contact forms and return contact URLs
        
        Args:
            domain: Website domain to crawl
            
        Returns:
            List of dictionaries with contact URL information
        """
        # Ensure domain has protocol
        if not domain.startswith(('http://', 'https://')):
            domain = f'https://{domain}'
            
        self.logger.info(f"🔍 Crawling {domain}")
        
        # Set domain for this crawl
        self.domain = domain
        self.base_domain = urlparse(domain).netloc
        self.found_forms = []
        self.crawled_urls = set()
        
        try:
            # Find contact forms using the original method
            await self.find_contact_forms()
            
            # Extract contact URLs
            contact_urls = []
            
            # First add URLs from found forms
            for form in self.found_forms:
                # Get URL of the page containing the form
                page_url = form.get("page_url", "")
                source_link = form.get("source_link", {})
                link_text = source_link.get("text", "Unknown")
                confidence = form.get("confidence_score", 0.5)
                
                # Add to results
                contact_urls.append({
                    "domain": domain,
                    "contact_url": page_url,
                    "link_text": link_text,
                    "confidence": confidence
                })
                
            # If no forms found, but we found contact links, include those
            if not contact_urls:
                # Get all links and classify them
                all_links = await self._get_all_links()
                contact_links = self._classify_links_with_model(all_links)
                
                for link in contact_links[:3]:  # Top 3 contact links
                    contact_urls.append({
                        "domain": domain,
                        "contact_url": link.get("full_url", ""),
                        "link_text": link.get("text", ""),
                        "confidence": link.get("confidence", 0.5)
                    })
                    
            # Remove duplicates
            seen_urls = set()
            unique_urls = []
            
            for item in contact_urls:
                if item["contact_url"] not in seen_urls:
                    seen_urls.add(item["contact_url"])
                    unique_urls.append(item)
                    
            return unique_urls
            
        except Exception as e:
            self.logger.error(f"Error processing {domain}: {str(e)}")
            return []

    def _save_results_to_csv(
        self,
        save_detail: bool = False,
        column_name: str = "contact_url",
        output_dir: str = "tmp"
    ) -> str:
        """
        Save crawling results to CSV file with only contact URLs

        Args:
            save_detail: If True, also save a detailed version with all columns
            column_name: Name for the contact URL column in the CSV
            output_dir: Optional output directory for the CSV

        Returns:
            Path to the generated CSV file
        """
        if not self.results:
            self.logger.warning("No results to save")
            return ""

        # Use provided output_dir if given
        out_dir = Path(output_dir) if output_dir else self.output_dir
        out_dir.mkdir(exist_ok=True)

        # Generate output filename with timestamp
        timestamp = int(time.time())
        csv_filename = f"contact_links_{timestamp}.csv"
        output_path = out_dir / csv_filename

        # Extract just the contact URLs
        contact_urls = [result['contact_url'] for result in self.results]

        # Create DataFrame with custom column name
        df = pd.DataFrame(contact_urls, columns=[column_name])

        # Save to CSV without index
        df.to_csv(output_path, index=False, header=True)

        # Only save detailed version if requested
        if save_detail:
            detail_filename = f"contact_links_detail_{timestamp}.csv"
            detail_path = out_dir / detail_filename
            pd.DataFrame(self.results).to_csv(detail_path, index=False)
            self.logger.info(f"Detailed version saved to {detail_path}")

        self.logger.info(f"Saved {len(contact_urls)} contact URLs to {output_path} with column name '{column_name}'")

        return str(output_path)

    # =========================================================================
    # Original methods below, kept intact
    # =========================================================================

    async def find_contact_forms(self) -> List[Dict]:
        """
        Two-stage process: ML classification then form detection
        """
        # Load models first to get accurate status
        self._load_pretrained_models()

        self.logger.info(
            f"\n"
            + "="*80
            + "\n🎯 TWO-STAGE CONTACT FORM FINDER\n"
            + "="*80
            + f"\nTarget URL: {self.domain}\n"
            + f"ML Model: {'✅ Pre-trained' if self.classifier else '❌ Using heuristics'}\n"
            + "="*80
        )

        # STAGE 1: Get all links and classify them
        self.logger.info(
            f"\n🔍 STAGE 1: LINK CLASSIFICATION\n"
            + "="*40
        )

        all_links = await self._get_all_links()
        contact_links = self._classify_links_with_model(all_links)

        if not contact_links:
            self.logger.warning("❌ No contact-related links found")
            return []

        # STAGE 2: Visit contact links and check for forms
        self.logger.info(
            f"\n🔍 STAGE 2: FORM DETECTION\n"
            + "="*40
        )

        actual_forms = await self._check_links_for_forms(contact_links)

        self.found_forms = actual_forms
        return actual_forms

    async def _get_all_links(self) -> List[Dict]:
        """Get all links from the main page"""
        self.logger.info(f"🌐 Fetching main page: {self.domain}")

        browser_config = BrowserConfig(headless=True, verbose=False)
        crawl_config = CrawlerRunConfig(
            wait_for_images=False, screenshot=False, pdf=False, verbose=False
        )

        async with AsyncWebCrawler(config=browser_config) as crawler:
            crawl_result = await crawler.arun(
                self.domain, config=crawl_config
            )

            if not crawl_result.success:
                raise Exception(f"Failed to crawl page: {crawl_result.error_message}")

            self.logger.info(f"✅ Page fetched (Status: {crawl_result.status_code})")

            soup = BeautifulSoup(crawl_result.html, "html.parser")
            links = soup.find_all("a", href=True)

            all_links = []
            for link in links:
                href = link["href"]
                text = link.get_text(strip=True)

                if not text or href.startswith(("javascript:", "mailto:", "tel:", "#")):
                    continue

                # Convert to absolute URL
                if href.startswith("/"):
                    full_url = urljoin(self.domain, href)
                elif href.startswith("http"):
                    full_url = href
                else:
                    full_url = urljoin(self.domain, href)

                # Only keep links from same domain
                if urlparse(full_url).netloc != self.base_domain:
                    continue

                all_links.append(
                    {
                        "text": text,
                        "href": href,
                        "full_url": full_url,
                        "context": self._get_link_context(link),
                    }
                )

            self.logger.info(f"✅ Found {len(all_links)} same-domain links")

            # Normalize and deduplicate links
            normalized_links = self._normalize_and_deduplicate_links(all_links)
            self.logger.info(f"✅ After deduplication: {len(normalized_links)} unique links")

            self.crawled_urls.add(self.domain)
            return normalized_links

    def _normalize_and_deduplicate_links(self, all_links: List[Dict]) -> List[Dict]:
        """Normalize URLs and deduplicate links while preserving best text/context"""
        from urllib.parse import urlparse, urlunparse
        import re

        # Group links by normalized URL
        url_groups = {}

        for link in all_links:
            # Normalize URL
            parsed = urlparse(link["full_url"])
            # Remove fragments, normalize path
            normalized_path = re.sub(r"/+", "/", parsed.path.rstrip("/") or "/")
            normalized_url = urlunparse(
                (
                    parsed.scheme,
                    parsed.netloc,
                    normalized_path,
                    parsed.params,
                    parsed.query,
                    "",  # Remove fragment
                )
            )

            if normalized_url not in url_groups:
                url_groups[normalized_url] = []
            url_groups[normalized_url].append(link)

        # For each group, pick the best representative
        deduplicated = []
        for normalized_url, link_group in url_groups.items():
            if len(link_group) == 1:
                deduplicated.append(link_group[0])
            else:
                # Pick best link: prefer longer/more descriptive text
                best_link = max(
                    link_group,
                    key=lambda link_item: (
                        len(link_item["text"].strip()),  # Longer text first
                        "contact"
                        in link_item["text"].lower(),  # Contact-related text preferred
                        len(link_item["context"].strip()),  # More context
                    ),
                )

                # Merge all text variants for better ML classification
                all_texts = [
                    link_item["text"]
                    for link_item in link_group
                    if link_item["text"].strip()
                ]
                all_contexts = [
                    link_item["context"]
                    for link_item in link_group
                    if link_item["context"].strip()
                ]

                best_link["text"] = " | ".join(set(all_texts))
                best_link["context"] = " | ".join(set(all_contexts))
                best_link["full_url"] = normalized_url  # Use normalized URL

                deduplicated.append(best_link)

        return deduplicated

    def _get_link_context(self, link_element) -> str:
        """Get surrounding context for a link"""
        try:
            parent = link_element.parent
            if parent:
                context = parent.get_text(strip=True)
                return context[:100] if len(context) > 100 else context
        except Exception:
            pass
        return ""

    def _classify_links_with_model(self, all_links: List[Dict]) -> List[Dict]:
        """Classify each link using the pre-trained model or heuristics"""
        self.logger.info(f"🤖 Classifying {len(all_links)} links...")

        if self.classifier is not None:
            return self._classify_with_pretrained_model(all_links)
        else:
            return self._classify_links_heuristic(all_links)

    def _classify_with_pretrained_model(self, all_links: List[Dict]) -> List[Dict]:
        """Use pre-trained model to classify links"""
        self.logger.info(f"   Using pre-trained binary classifier...")

        contact_links = []

        # Get all link texts for batch processing
        link_texts = [f"{link['text']} {link['context']}" for link in all_links]

        # Get embeddings for all links
        embeddings = self.sentence_model.encode(link_texts)

        # Get predictions and probabilities
        probabilities = self.classifier.predict_proba(embeddings)

        for i, (link, prob) in enumerate(zip(all_links, probabilities), 1):
            confidence = float(prob[1])  # Probability of being contact-related
            is_contact = confidence > 0.25

            status = "✅ YES" if is_contact else "❌ NO"
            self.logger.info(f"   {i:2d}. {status} ({confidence:.3f}) {link['full_url']}")

            if is_contact:
                link["confidence"] = confidence
                link["ml_prediction"] = "YES"
                link["classifier_confidence"] = confidence
                contact_links.append(link)

        self.logger.info(
            f"📊 Classification Results:\n"
            f"   - Total links: {len(all_links)}\n"
            f"   - Contact links (YES): {len(contact_links)}\n"
            f"   - Non-contact links (NO): {len(all_links) - len(contact_links)}"
        )

        return contact_links

    def _classify_links_heuristic(self, all_links: List[Dict]) -> List[Dict]:
        """Fallback heuristic classification"""
        self.logger.info(f"   Using heuristic classification...")

        contact_links = []
        contact_keywords = ["contact", "support", "help", "about", "feedback", "sales"]

        for i, link in enumerate(all_links, 1):
            text_lower = f"{link['text']} {link['context']}".lower()

            is_contact = any(keyword in text_lower for keyword in contact_keywords)
            confidence = 0.7 if is_contact else 0.3

            status = "✅ YES" if is_contact else "❌ NO"
            self.logger.info(f"   {i:2d}. {status} ({confidence:.3f}) '{link['text'][:40]}...' → {link['href']}")

            if is_contact:
                link["confidence"] = confidence
                link["ml_prediction"] = "YES"
                contact_links.append(link)

        self.logger.info(
            f"📊 Classification Results:\n"
            f"   - Total links: {len(all_links)}\n"
            f"   - Contact links (YES): {len(contact_links)}"
        )

        return contact_links

    async def _check_links_for_forms(self, contact_links: List[Dict]) -> List[Dict]:
        """Visit each contact link and check for actual contact forms"""
        self.logger.info(f"🔍 Checking {len(contact_links)} contact links for forms...")

        browser_config = BrowserConfig(headless=True, verbose=False)
        crawl_config = CrawlerRunConfig(
            wait_for_images=False, screenshot=False, pdf=False, verbose=False
        )

        actual_forms = []

        async with AsyncWebCrawler(config=browser_config) as crawler:
            for i, link in enumerate(contact_links, 1):
                self.logger.info(f"   {i}. Checking: {link['full_url']}")

                urls_to_check = self._generate_url_variations(link["full_url"])

                form_found = False
                for j, url_variant in enumerate(urls_to_check):
                    if url_variant in self.crawled_urls:
                        continue

                    variation_label = f"main" if j == 0 else f"variant {j}"
                    self.logger.info(f"      🔗 Trying {variation_label}: {url_variant}")

                    try:
                        crawl_result = await crawler.arun(
                            url_variant, config=crawl_config
                        )

                        if not crawl_result.success:
                            self.logger.warning(
                                f"         ❌ Failed to load (Status: {crawl_result.status_code if crawl_result.status_code else 'No response'})"
                            )
                            continue

                        self.logger.info(f"         ✅ Page loaded (Status: {crawl_result.status_code})")

                        soup = BeautifulSoup(crawl_result.html, "html.parser")
                        forms = self._find_contact_forms_on_page(soup, url_variant)

                        if forms:
                            self.logger.info(
                                f"         🎯 Found {len(forms)} contact form(s)!"
                            )
                            for form in forms:
                                form["source_link"] = link
                                form["actual_url"] = url_variant
                                form["link_confidence"] = link.get(
                                    "classifier_confidence", link.get("confidence", 0)
                                )
                                actual_forms.append(form)
                            form_found = True
                        else:
                            self.logger.info(f"         ➡️ No forms found on this variant")

                        self.crawled_urls.add(url_variant)

                        if form_found:
                            break

                        await asyncio.sleep(0.5)

                    except Exception as e:
                        self.logger.error(f"         ❌ Error checking variant: {str(e)}")

                if not form_found:
                    self.logger.warning(f"      ❌ No forms found in any variation")

                await asyncio.sleep(1)

        deduplicated_forms = self._deduplicate_forms(actual_forms)

        self.logger.info(
            f"📊 Form Detection Results:\n"
            f"   - Contact links checked: {len(contact_links)}\n"
            f"   - Total forms found: {len(actual_forms)}\n"
            f"   - Unique forms (after deduplication): {len(deduplicated_forms)}"
        )

        return deduplicated_forms

    def _deduplicate_forms(self, forms: List[Dict]) -> List[Dict]:
        """Remove duplicate forms based on form signature"""
        if not forms:
            return forms

        seen_signatures = set()
        unique_forms = []

        for form in forms:
            # Create a signature for the form
            if form.get("form_type") == "iframe_form":
                # For iframe forms, use cleaned iframe src as signature
                iframe_src = form.get("iframe_src", "")
                # Remove query parameters that might be page-specific
                if "?" in iframe_src:
                    signature = iframe_src.split("?")[0]
                else:
                    signature = iframe_src
            else:
                # For HTML forms, use action + method + field types
                action = form.get("form_action", "")
                method = form.get("form_method", "GET")
                field_types = sorted(form.get("field_types", {}).keys())
                signature = f"{action}|{method}|{'|'.join(field_types)}"

            if signature not in seen_signatures:
                seen_signatures.add(signature)
                # Add info about all pages where this form was found
                form["found_on_pages"] = [form["page_url"]]
                unique_forms.append(form)
            else:
                # Find the existing form and add this page to its list
                for existing_form in unique_forms:
                    existing_signature = ""
                    if existing_form.get("form_type") == "iframe_form":
                        existing_src = existing_form.get("iframe_src", "")
                        if "?" in existing_src:
                            existing_signature = existing_src.split("?")[0]
                        else:
                            existing_signature = existing_src
                    else:
                        existing_action = existing_form.get("form_action", "")
                        existing_method = existing_form.get("form_method", "GET")
                        existing_field_types = sorted(
                            existing_form.get("field_types", {}).keys()
                        )
                        existing_signature = f"{existing_action}|{existing_method}|{'|'.join(existing_field_types)}"

                    if existing_signature == signature:
                        if "found_on_pages" not in existing_form:
                            existing_form["found_on_pages"] = [
                                existing_form["page_url"]
                            ]
                        existing_form["found_on_pages"].append(form["page_url"])
                        break

        # Update the display info for unique forms
        for form in unique_forms:
            if "found_on_pages" in form and len(form["found_on_pages"]) > 1:
                form["duplicate_count"] = len(form["found_on_pages"])
                self.logger.info(f"   📋 Deduplicated: Same form found on {len(form['found_on_pages'])} pages")

        return unique_forms

    def _find_contact_forms_on_page(
        self, soup: BeautifulSoup, page_url: str
    ) -> List[Dict]:
        """Find actual contact forms on a page (including iframe forms)"""
        contact_forms = []

        # 1. Check for traditional HTML forms
        html_forms = self._find_html_forms(soup, page_url)
        contact_forms.extend(html_forms)

        # 2. Check for iframe forms (embedded contact forms)
        iframe_forms = self._find_iframe_forms(soup, page_url)
        contact_forms.extend(iframe_forms)

        return contact_forms

    def _find_html_forms(self, soup: BeautifulSoup, page_url: str) -> List[Dict]:
        """Find traditional HTML forms on the page"""
        forms = soup.find_all("form")
        contact_forms = []

        for i, form in enumerate(forms):
            # Analyze form fields
            inputs = form.find_all(["input", "textarea", "select"])

            field_types = {}
            contact_field_count = 0

            for inp in inputs:
                input_type = inp.get("type", "text").lower()
                name = inp.get("name", "").lower()
                placeholder = inp.get("placeholder", "").lower()

                # Classify field type
                if input_type == "email" or "email" in name or "email" in placeholder:
                    field_types["email"] = field_types.get("email", 0) + 1
                    contact_field_count += 1
                elif inp.name == "textarea" or "message" in name or "comment" in name:
                    field_types["message"] = field_types.get("message", 0) + 1
                    contact_field_count += 1
                elif "name" in name or "name" in placeholder:
                    field_types["name"] = field_types.get("name", 0) + 1
                    contact_field_count += 1
                elif "phone" in name or "phone" in placeholder:
                    field_types["phone"] = field_types.get("phone", 0) + 1
                    contact_field_count += 1

            # Consider it a contact form if it has at least 2 contact-related fields
            if contact_field_count >= 2:
                confidence = min(0.5 + (contact_field_count * 0.15), 1.0)

                contact_forms.append(
                    {
                        "page_url": page_url,
                        "form_index": i,
                        "form_type": "html_form",
                        "field_types": field_types,
                        "field_count": len(inputs),
                        "contact_field_count": contact_field_count,
                        "confidence_score": confidence,
                        "is_contact_form": True,
                        "form_action": form.get("action", ""),
                        "form_method": form.get("method", "GET").upper(),
                        "discovery_method": "html_form_detection",
                    }
                )

        return contact_forms

    def _find_iframe_forms(self, soup: BeautifulSoup, page_url: str) -> List[Dict]:
        """Find iframe-based contact forms"""
        iframes = soup.find_all("iframe")
        contact_forms = []

        # Known contact form service patterns
        contact_form_services = {
            "typeform.com": "Typeform",
            "forms.gle": "Google Forms",
            "google.com/forms": "Google Forms",
            "jotform.com": "JotForm",
            "wufoo.com": "Wufoo",
            "formstack.com": "Formstack",
            "hubspot.com": "HubSpot",
            "mailchimp.com": "Mailchimp",
            "constantcontact.com": "Constant Contact",
            "formspree.io": "Formspree",
            "netlify.com": "Netlify Forms",
            "calendly.com": "Calendly",
            "acuityscheduling.com": "Acuity Scheduling",
            "tally.so": "Tally",
            "airtable.com": "Airtable Forms",
        }

        for i, iframe in enumerate(iframes):
            src = iframe.get("src", "")
            if not src:
                continue

            # Check if iframe contains a known contact form service
            detected_service = None
            for service_domain, service_name in contact_form_services.items():
                if service_domain in src.lower():
                    detected_service = service_name
                    break

            # Also check for common contact-related keywords in iframe src
            contact_keywords = [
                "contact", "form", "inquiry", "quote", "appointment", "booking",
            ]
            has_contact_keywords = any(keyword in src.lower() for keyword in contact_keywords)

            # Check iframe attributes for contact-related info
            iframe_class = iframe.get("class", [])
            iframe_id = iframe.get("id", "")
            iframe_title = iframe.get("title", "")

            if isinstance(iframe_class, list):
                iframe_class = " ".join(iframe_class)

            iframe_attrs = f"{iframe_class} {iframe_id} {iframe_title}".lower()
            has_contact_attrs = any(keyword in iframe_attrs for keyword in contact_keywords)

            if detected_service or has_contact_keywords or has_contact_attrs:
                # Calculate confidence based on detection method
                confidence = 0.6  # Base confidence for iframe forms
                if detected_service:
                    confidence += 0.3  # Higher confidence for known services
                if has_contact_keywords:
                    confidence += 0.2
                if has_contact_attrs:
                    confidence += 0.1

                confidence = min(confidence, 1.0)

                contact_forms.append({
                    "page_url": page_url,
                    "form_index": i,
                    "form_type": "iframe_form",
                    "iframe_src": src,
                    "detected_service": detected_service or "Unknown",
                    "confidence_score": confidence,
                    "is_contact_form": True,
                    "form_action": src,
                    "form_method": "IFRAME",
                    "discovery_method": "iframe_form_detection",
                    "field_types": {"iframe": 1},  # Placeholder since we can't inspect iframe content
                    "field_count": 1,
                    "contact_field_count": 1,
                })

                self.logger.info(f"            🎯 Found iframe contact form: {detected_service or 'Unknown service'}")
                self.logger.info(f"               Source: {src[:80]}{'...' if len(src) > 80 else ''}")

        return contact_forms

    def _generate_url_variations(self, base_url: str) -> List[str]:
        """Generate common URL variations to check for contact forms"""
        from urllib.parse import urlparse, urlunparse

        variations = [base_url]  # Always try the original first

        parsed = urlparse(base_url)
        path = parsed.path.rstrip("/")

        # Common file variations for contact pages
        common_files = [
            "index.html",
            "index.php", 
            "default.html",
            "contact.html",
            "contact.php",
        ]

        for file in common_files:
            # Add file to the path
            if path.endswith(".html") or path.endswith(".php"):
                # Replace existing file
                path_parts = path.split("/")
                path_parts[-1] = file
                new_path = "/".join(path_parts)
            else:
                # Add file to directory path
                new_path = f"{path}/{file}" if path else f"/{file}"

            variation_url = urlunparse(
                (
                    parsed.scheme,
                    parsed.netloc,
                    new_path,
                    parsed.params,
                    parsed.query,
                    parsed.fragment,
                )
            )

            if variation_url not in variations:
                variations.append(variation_url)

        # If original doesn't end with /, try with trailing slash
        if not base_url.endswith("/") and not any(
            base_url.endswith(ext) for ext in [".html", ".php", ".asp", ".aspx"]
        ):
            variations.append(f"{base_url}/")

        return variations


# CLI interface for direct usage
async def main():
    """CLI entry point for contact form crawler"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Contact Form Crawler")
    parser.add_argument(
        "--csv", 
        type=str,
        default="live_test_sample.csv",
        help="CSV file containing domains to crawl (default: live_test_sample.csv)"
    )
    parser.add_argument(
        "--output-dir", 
        type=str,
        default="tmp",
        help="Output directory for results (default: tmp)"
    )
    parser.add_argument(
        "--detail",
        action="store_true",
        help="Save detailed CSV with all fields (domain, link text, confidence, etc.)"
    )
    parser.add_argument(
        "--columnName",
        type=str,
        default="contact_url",
        help="Name for the contact URL column in the CSV (default: contact_url)"
    )
    
    args = parser.parse_args()
    
    # Validate CSV file
    csv_path = args.csv
    if not Path(csv_path).exists():
        # Use a temporary logger for this error
        logger = get_logger()
        logger.error(f"❌ CSV file not found: {csv_path}")
        return 1
        
    # Create crawler and process domains
    crawler = ContactFormCrawler(
        csv_file_path=csv_path,
        output_dir=args.output_dir
    )
    
    try:
        # Process CSV
        output_path = await crawler.crawl_csv_domains(save_detail=args.detail, column_name=args.columnName)
        
        crawler.logger.info(
            "✅ Crawling completed successfully!\n"
            f"📄 Results saved to: {output_path}\n"
            f"📝 Column name: '{args.columnName}'"
        )
        if args.detail:
            detail_path = output_path.replace("contact_links_", "contact_links_detail_")
            crawler.logger.info(f"📊 Detailed results saved to: {detail_path}")
        
        if crawler.results:
            crawler.logger.info(
                "📊 Summary:\n"
                f"   - Total contact URLs found: {len(crawler.results)}\n"
                f"   - Domains with contact URLs: {len(set(r['domain'] for r in crawler.results))}"
            )
    
        return 0
        
    except Exception as e:
        crawler.logger.error(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))
