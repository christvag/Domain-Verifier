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

    args = parser.parse_args()

    # Validate URL
    if not args.url.startswith(("http://", "https://")):
        args.url = "https://" + args.url

    # Test the crawler
    crawler = ContactFormCrawler(args.url)

    try:
        contact_links = await crawler.find_contact_forms()

        # Export results
        filename = crawler.export_results()
        print(f"\n💾 Results exported to: {filename}")

        # Exit with appropriate code
        if contact_links:
            print(f"\n✅ SUCCESS: Found {len(contact_links)} contact links")
            sys.exit(0)
        else:
            print(f"\n❌ No contact links found")
            sys.exit(2)

    except Exception as e:
        print(f"\n❌ FAILED: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
