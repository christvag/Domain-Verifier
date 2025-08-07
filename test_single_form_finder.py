#!/usr/bin/env python3
"""
Test the Navigation Contact Finder

Usage:
    python test_single_form_finder.py https://example.com
    python test_single_form_finder.py  # Uses default test site
"""

import asyncio
import argparse
import sys
import json
import time
from pathlib import Path
from urllib.parse import urlparse

from contact_form_crawler import ContactFormCrawler


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Test Navigation Contact Finder")
    parser.add_argument(
        "url",
        nargs="?",
        default="https://aldcoair.com",
        help="URL to test (default: https://aldcoair.com)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set logging level (default: INFO)"
    )

    args = parser.parse_args()

    # Set logger level globally
    import centralized_logger
    centralized_logger.get_logger().setLevel(args.log_level.upper())

    # Validate URL
    if not args.url.startswith(("http://", "https://")):
        args.url = "https://" + args.url

    print(f"🚀 Testing Contact Form Finder for: {args.url}")
    print("=" * 60)

    # Test the crawler
    crawler = ContactFormCrawler(domain=args.url)

    try:
        # Set the base_domain attribute that's needed for single domain mode
        crawler.base_domain = urlparse(args.url).netloc

        # Find contact forms
        contact_forms = await crawler.find_contact_forms()

        # Show results
        if contact_forms:
            print(f"\n✅ SUCCESS: Found {len(contact_forms)} contact form(s)")

            for i, form in enumerate(contact_forms, 1):
                print(f"\n🔗 Contact Form #{i}:")
                print(f"   📍 Page URL: {form.get('page_url', 'Unknown')}")
                print(f"   📋 Form Type: {form.get('form_type', 'Unknown')}")
                print(f"   🎯 Confidence: {form.get('confidence_score', 0):.2f}")

                if form.get("form_type") == "html_form":
                    print(f"   📝 Fields: {form.get('field_types', {})}")
                    print(f"   🔧 Action: {form.get('form_action', 'None')}")
                    print(f"   📨 Method: {form.get('form_method', 'Unknown')}")
                elif form.get("form_type") == "iframe_form":
                    print(f"   🔗 Service: {form.get('detected_service', 'Unknown')}")
                    print(f"   📎 Source: {form.get('iframe_src', 'Unknown')}")

                # Show source link info if available
                source_link = form.get("source_link", {})
                if source_link:
                    print(f"   🔍 Found via link: '{source_link.get('text', 'Unknown')}'")
                    print(f"   🎲 Link confidence: {source_link.get('confidence', 0):.2f}")

        # Export results to JSON file for compatibility
        results_dir = Path("tmp")
        results_dir.mkdir(exist_ok=True)

        timestamp = int(time.time())
        filename = results_dir / f"contact_forms_{timestamp}.json"

        # Prepare export data
        export_data = {
            "domain": args.url,
            "crawl_timestamp": timestamp,
            "total_forms_found": len(contact_forms),
            "forms": contact_forms,
        }

        # Save to JSON
        with open(filename, "w") as f:
            json.dump(export_data, f, indent=2, default=str)

        print(f"\n💾 Results exported to: {filename}")

        # Exit with appropriate code
        if contact_forms:
            print(f"\n✅ SUCCESS: Found {len(contact_forms)} contact form(s)")
            sys.exit(0)
        else:
            print(f"\n❌ No contact forms found")
            sys.exit(2)

    except Exception as e:
        print(f"\n❌ FAILED: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
