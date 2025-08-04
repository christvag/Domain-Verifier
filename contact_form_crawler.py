#!/usr/bin/env python3
"""
Two-Stage ML Contact Form Finder with Pre-trained Binary Classifier

Stage 1: Load pre-trained binary classifier for links (YES/NO for contact-related)
Stage 2: Visit YES links and check for actual contact forms
"""

import os
import warnings

# Enable tokenizer parallelism but limit it to avoid forking issues
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["OMP_NUM_THREADS"] = "1"  # Limit OpenMP threads

# Suppress transformer deprecation warnings
warnings.filterwarnings(
    "ignore", message=".*encoder_attention_mask.*", category=FutureWarning
)

import json
import time
import asyncio
import pickle
from typing import List, Dict
from urllib.parse import urljoin, urlparse, ParseResult
from pathlib import Path

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CrawlResult
from bs4 import BeautifulSoup
from centralized_logger import get_logger

try:
    from sentence_transformers import SentenceTransformer

    TRANSFORMER_AVAILABLE = True
except ImportError:
    TRANSFORMER_AVAILABLE = False


class ContactFormCrawler:
    """
    Two-stage contact form finder with pre-trained binary classifier:
    1. Load pre-trained binary classifier for link classification (YES/NO)
    2. Form detection on YES links
    """

    def __init__(self, domain: str):
        self.domain = domain.rstrip("/")
        self.base_domain = urlparse(domain).netloc
        self.logger = get_logger()

        # Results storage
        self.found_forms = []
        self.crawled_urls = set()

        # Initialize ML components
        self.sentence_model = None
        self.classifier = None
        self.model_dir = Path("models")
        self._models_loaded = False

    def _load_pretrained_models(self):
        """Load pre-trained models from disk"""
        if self._models_loaded:
            return

        if not TRANSFORMER_AVAILABLE:
            print("⚠️ sentence-transformers not available, using heuristics")
            return

        try:
            # Check if models exist
            sentence_model_path = self.model_dir / "sentence_model"
            classifier_path = self.model_dir / "contact_classifier.pkl"

            if not sentence_model_path.exists() or not classifier_path.exists():
                print(
                    "⚠️ Pre-trained models not found. Run contact_link_trainer.py first"
                )
                print(f"   Looking for: {sentence_model_path} and {classifier_path}")
                return

            print("📥 Loading pre-trained models...")

            # Load sentence transformer
            self.sentence_model = SentenceTransformer(str(sentence_model_path))

            # Load classifier
            with open(classifier_path, "rb") as f:
                self.classifier = pickle.load(f)

            print("✅ Pre-trained models loaded successfully")
            self._models_loaded = True

        except Exception as e:
            print(f"⚠️ Failed to load pre-trained models: {e}")
            print("   Run: python contact_link_trainer.py")
            self.sentence_model = None
            self.classifier = None

    async def find_contact_forms(self) -> List[Dict]:
        """
        Two-stage process: ML classification then form detection
        """
        # Load models first to get accurate status
        self._load_pretrained_models()

        print(f"\n{'='*80}")
        print("🎯 TWO-STAGE CONTACT FORM FINDER")
        print(f"{'='*80}")
        print(f"Target URL: {self.domain}")
        print(
            f"ML Model: {'✅ Pre-trained' if self.classifier else '❌ Using heuristics'}"
        )
        print(f"{'='*80}\n")

        # STAGE 1: Get all links and classify them
        print("🔍 STAGE 1: LINK CLASSIFICATION")
        print(f"{'='*40}")

        all_links = await self._get_all_links()
        contact_links = self._classify_links_with_model(all_links)

        if not contact_links:
            print("❌ No contact-related links found")
            return []

        # STAGE 2: Visit contact links and check for forms
        print("\n🔍 STAGE 2: FORM DETECTION")
        print(f"{'='*40}")

        actual_forms = await self._check_links_for_forms(contact_links)

        self.found_forms = actual_forms
        return actual_forms

    async def _get_all_links(self) -> List[Dict]:
        """Get all links from the main page"""
        print(f"🌐 Fetching main page: {self.domain}")

        browser_config = BrowserConfig(headless=True, verbose=False)
        crawl_config = CrawlerRunConfig(
            wait_for_images=False, screenshot=False, pdf=False, verbose=False
        )

        async with AsyncWebCrawler(config=browser_config) as crawler:
            crawl_result: CrawlResult = await crawler.arun(
                self.domain, config=crawl_config
            )

            if not crawl_result.success:
                raise Exception(f"Failed to crawl page: {crawl_result.error_message}")

            print(f"✅ Page fetched (Status: {crawl_result.status_code})")

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

            print(f"✅ Found {len(all_links)} same-domain links")

            # Normalize and deduplicate links
            normalized_links = self._normalize_and_deduplicate_links(all_links)
            print(f"✅ After deduplication: {len(normalized_links)} unique links")

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
            parsed: ParseResult = urlparse(link["full_url"])
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
        print(f"🤖 Classifying {len(all_links)} links...")

        if self.classifier is not None:
            return self._classify_with_pretrained_model(all_links)
        else:
            return self._classify_links_heuristic(all_links)

    def _classify_with_pretrained_model(self, all_links: List[Dict]) -> List[Dict]:
        """Use pre-trained model to classify links"""
        print("   Using pre-trained binary classifier...")

        contact_links = []

        # Get all link texts for batch processing
        link_texts = [f"{link['text']} {link['context']}" for link in all_links]

        # Get embeddings for all links
        embeddings = self.sentence_model.encode(link_texts)

        # Get predictions and probabilities
        probabilities = self.classifier.predict_proba(embeddings)

        for i, (link, prob) in enumerate(zip(all_links, probabilities), 1):
            # Use threshold of 0.25 for contact classification (lowered to catch more cases)
            confidence = float(prob[1])  # Probability of being contact-related
            is_contact = confidence > 0.25

            # Show individual prediction
            status = "✅ YES" if is_contact else "❌ NO"
            print(f"   {i:2d}. {status} ({confidence:.3f}) {link['full_url']}")

            if is_contact:
                link["confidence"] = confidence
                link["ml_prediction"] = "YES"
                link["classifier_confidence"] = confidence
                contact_links.append(link)

        print("\n📊 Classification Results:")
        print(f"   - Total links: {len(all_links)}")
        print(f"   - Contact links (YES): {len(contact_links)}")
        print(f"   - Non-contact links (NO): {len(all_links) - len(contact_links)}")

        return contact_links

    def _classify_links_heuristic(self, all_links: List[Dict]) -> List[Dict]:
        """Fallback heuristic classification"""
        print("   Using heuristic classification...")

        contact_links = []
        contact_keywords = ["contact", "support", "help", "about", "feedback", "sales"]

        for i, link in enumerate(all_links, 1):
            text_lower = f"{link['text']} {link['context']}".lower()

            is_contact = any(keyword in text_lower for keyword in contact_keywords)
            confidence = 0.7 if is_contact else 0.3

            status = "✅ YES" if is_contact else "❌ NO"
            print(
                f"   {i:2d}. {status} ({confidence:.3f}) '{link['text'][:40]}...' → {link['href']}"
            )

            if is_contact:
                link["confidence"] = confidence
                link["ml_prediction"] = "YES"
                contact_links.append(link)

        print("\n📊 Classification Results:")
        print(f"   - Total links: {len(all_links)}")
        print(f"   - Contact links (YES): {len(contact_links)}")

        return contact_links

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

    async def _check_links_for_forms(self, contact_links: List[Dict]) -> List[Dict]:
        """Visit each contact link and check for actual contact forms"""
        print(f"🔍 Checking {len(contact_links)} contact links for forms...")

        browser_config = BrowserConfig(headless=True, verbose=False)
        crawl_config = CrawlerRunConfig(
            wait_for_images=False, screenshot=False, pdf=False, verbose=False
        )

        actual_forms = []

        async with AsyncWebCrawler(config=browser_config) as crawler:
            for i, link in enumerate(contact_links, 1):
                print(f"\n   {i}. Checking: {link['full_url']}")

                # Generate URL variations to check
                urls_to_check = self._generate_url_variations(link["full_url"])

                form_found = False
                for j, url_variant in enumerate(urls_to_check):
                    if url_variant in self.crawled_urls:
                        continue  # Skip already crawled URLs

                    variation_label = f"main" if j == 0 else f"variant {j}"
                    print(f"      🔗 Trying {variation_label}: {url_variant}")

                    try:
                        # Visit the page
                        crawl_result = await crawler.arun(
                            url_variant, config=crawl_config
                        )

                        if not crawl_result.success:
                            print(
                                f"         ❌ Failed to load (Status: {crawl_result.status_code if crawl_result.status_code else 'No response'})"
                            )
                            continue

                        print(
                            f"         ✅ Page loaded (Status: {crawl_result.status_code})"
                        )

                        # Parse and look for forms
                        soup = BeautifulSoup(crawl_result.html, "html.parser")
                        forms = self._find_contact_forms_on_page(soup, url_variant)

                        if forms:
                            print(f"         🎯 Found {len(forms)} contact form(s)!")
                            for form in forms:
                                form["source_link"] = link
                                form["actual_url"] = (
                                    url_variant  # Track which variation worked
                                )
                                form["link_confidence"] = link.get(
                                    "classifier_confidence", link.get("confidence", 0)
                                )
                                actual_forms.append(form)
                            form_found = True
                        else:
                            print(f"         ➡️ No forms found on this variant")

                        self.crawled_urls.add(url_variant)

                        # If we found forms, no need to check more variations for this link
                        if form_found:
                            break

                        # Be polite - wait between requests
                        await asyncio.sleep(0.5)

                    except Exception as e:
                        print(f"         ❌ Error checking variant: {str(e)}")

                if not form_found:
                    print(f"      ❌ No forms found in any variation")

                # Longer wait between different contact links
                await asyncio.sleep(1)

        # Deduplicate forms based on form signature
        deduplicated_forms = self._deduplicate_forms(actual_forms)

        print("\n📊 Form Detection Results:")
        print(f"   - Contact links checked: {len(contact_links)}")
        print(f"   - Total forms found: {len(actual_forms)}")
        print(f"   - Unique forms (after deduplication): {len(deduplicated_forms)}")

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
                print(
                    f"   📋 Deduplicated: Same form found on {len(form['found_on_pages'])} pages"
                )

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
                "contact",
                "form",
                "inquiry",
                "quote",
                "appointment",
                "booking",
            ]
            has_contact_keywords = any(
                keyword in src.lower() for keyword in contact_keywords
            )

            # Check iframe attributes for contact-related info
            iframe_class = iframe.get("class", [])
            iframe_id = iframe.get("id", "")
            iframe_title = iframe.get("title", "")

            if isinstance(iframe_class, list):
                iframe_class = " ".join(iframe_class)

            iframe_attrs = f"{iframe_class} {iframe_id} {iframe_title}".lower()
            has_contact_attrs = any(
                keyword in iframe_attrs for keyword in contact_keywords
            )

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

                contact_forms.append(
                    {
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
                        "field_types": {
                            "iframe": 1
                        },  # Placeholder since we can't inspect iframe content
                        "field_count": 1,
                        "contact_field_count": 1,
                    }
                )

                print(
                    f"            🎯 Found iframe contact form: {detected_service or 'Unknown service'}"
                )
                print(
                    f"               Source: {src[:80]}{'...' if len(src) > 80 else ''}"
                )

        return contact_forms

    def get_best_form(self):
        """Get the best contact form found"""
        if not self.found_forms:
            return None
        return max(self.found_forms, key=lambda f: f["confidence_score"])

    def export_results(self, filename=None):
        """Export results to JSON file"""
        if filename is None:
            domain_clean = self.base_domain.replace(".", "_").replace(":", "_")
            filename = f"contact_forms_{domain_clean}_{int(time.time())}.json"

        results = {
            "domain": self.domain,
            "timestamp": time.time(),
            "pretrained_model_used": self.classifier is not None,
            "total_forms_found": len(self.found_forms),
            "total_pages_crawled": len(self.crawled_urls),
            "best_form": self.get_best_form(),
            "all_forms": self.found_forms,
            "crawled_urls": list(self.crawled_urls),
            "method": "pretrained_binary_classifier",
        }

        with open(filename, "w") as f:
            json.dump(results, f, indent=2, default=str)

        return filename
