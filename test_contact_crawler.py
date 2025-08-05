#!/usr/bin/env python3
"""
Test Contact Form Crawler with CSV input

Usage:
    python test_contact_crawler.py                                    # Uses live_test_sample.csv
    python test_contact_crawler.py --csv my_domains.csv              # Uses custom CSV
    python test_contact_crawler.py --csv my_domains.csv --limit 5    # Custom CSV with limit
    python test_contact_crawler.py --limit 10 --output-dir test_results
"""

import asyncio
import sys
import argparse
from pathlib import Path
from contact_form_crawler import ContactFormCrawler


class LimitedContactFormCrawler(ContactFormCrawler):
    """Extended crawler with result limit functionality and real-time output"""
    
    def __init__(self, csv_file_path: str = None, output_dir: str = "contact_results", result_limit: int = None):
        super().__init__(csv_file_path=csv_file_path, output_dir=output_dir)
        self.result_limit = result_limit
        self.stop_crawling = False
        self.current_count = 0  # Track the current count for real-time display
    
    async def _crawl_single_domain(self, domain: str) -> list:
        """Override to check result limit and show results in real-time"""
        if self.stop_crawling:
            self.logger.info(f"⏹️ Stopping crawl - result limit reached")
            return []
        
        # Call parent method
        results = await super()._crawl_single_domain(domain)
        
        # Print results as they're found (for debugging)
        if results:
            print(f"\n🔍 Found {len(results)} contact URLs for {domain}:")
            for i, result in enumerate(results, 1):
                self.current_count += 1
                print(f"   #{self.current_count}: {result['contact_url']}")
                print(f"      Text: '{result['link_text']}'")
                print(f"      Confidence: {result['confidence']:.2f}")
                
                # If we have a limit, show progress toward it
                if self.result_limit:
                    print(f"      Progress: {self.current_count}/{self.result_limit}")
                    
                # Add separator between results
                if i < len(results):
                    print("      ---")
        
        # Check if we've hit the limit
        if self.result_limit and len(self.results) + len(results) >= self.result_limit:
            # Trim results to exact limit
            remaining_slots = self.result_limit - len(self.results)
            if remaining_slots > 0:
                results = results[:remaining_slots]
            self.stop_crawling = True
            self.logger.info(f"🎯 Result limit of {self.result_limit} reached. Stopping crawl.")
            print(f"\n🎯 LIMIT REACHED: Found {self.result_limit} contact URLs. Stopping crawl.")
        
        return results
    
    async def crawl_csv_domains(self, save_detail: bool = False) -> str:
        """Override to handle early stopping and show progress"""
        if not self.csv_file_path or not Path(self.csv_file_path).exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_file_path}")

        # Read CSV file
        import pandas as pd
        df = pd.read_csv(self.csv_file_path)
        
        # Check for website column
        if 'website' not in df.columns:
            raise ValueError("CSV must have a 'website' column")

        domains = df['website'].dropna().tolist()
        
        if self.result_limit:
            self.logger.info(f"🚀 Starting limited crawl: {len(domains)} domains (stopping at {self.result_limit} results)")
            print(f"🚀 Starting crawl with limit of {self.result_limit} contact URLs")
        else:
            self.logger.info(f"🚀 Starting full crawl: {len(domains)} domains")
            print(f"🚀 Starting full crawl of {len(domains)} domains")

        print("=" * 60)
        print("Real-time results will appear below as they're discovered:")
        print("=" * 60)
        
        # Process domains with concurrency control
        semaphore = asyncio.Semaphore(5)  # Limit to 5 concurrent crawls
        
        # Process domains one by one to check limit after each
        for i, domain in enumerate(domains):
            if self.stop_crawling:
                self.logger.info(f"⏹️ Stopped early after processing {i} domains")
                break
                
            domain = domain.strip()
            if not domain:
                continue
                
            print(f"\n📌 Processing domain {i+1}/{len(domains)}: {domain}")
            
            try:
                result = await self._crawl_single_domain_with_semaphore(domain, semaphore)
                if result:
                    self.results.extend(result)
                    
                # Log progress
                if self.result_limit:
                    self.logger.info(f"📊 Progress: {len(self.results)}/{self.result_limit} results found")
                else:
                    self.logger.info(f"📊 Progress: {len(self.results)} results found from {i+1}/{len(domains)} domains")
                    
            except Exception as e:
                self.logger.error(f"Error crawling {domain}: {e}")
                print(f"❌ Error processing {domain}: {str(e)}")

        # Save results to CSV
        output_path = self._save_results_to_csv(save_detail=save_detail)
        
        self.logger.info(f"✅ Crawling completed. Found {len(self.results)} contact URLs")
        self.logger.info(f"📄 Results saved to: {output_path}")
        
        return output_path


async def test_crawler(csv_file: str = "live_test_sample.csv", result_limit: int = None, 
                       output_dir: str = "contact_results", save_detail: bool = False):
    """Test the contact form crawler with the specified CSV"""
    
    if not Path(csv_file).exists():
        print(f"❌ CSV file not found: {csv_file}")
        print("Please ensure the CSV file exists and has a 'website' column")
        return False
    
    print(f"🚀 Testing Contact Form Crawler with {csv_file}")
    if result_limit:
        print(f"🎯 Result limit: {result_limit} contact URLs")
    print("=" * 60)
    
    # Initialize limited crawler
    crawler = LimitedContactFormCrawler(
        csv_file_path=csv_file,
        output_dir=output_dir,
        result_limit=result_limit
    )
    
    try:
        # Show CSV info
        import pandas as pd
        df = pd.read_csv(csv_file)
        print(f"📊 CSV contains {len(df)} domains")
        print(f"🔍 Sample domains: {df['website'].head(3).tolist()}")
        
        if result_limit and result_limit < len(df):
            print(f"⚠️  Will stop after finding {result_limit} contact URLs (may process fewer than {len(df)} domains)")
        
        # Run crawler
        output_path = await crawler.crawl_csv_domains(save_detail=save_detail)
        
        print(f"\n✅ Crawling completed!")
        print(f"📄 Results saved to: {output_path}")
        if save_detail:
            detail_path = output_path.replace("contact_links_", "contact_links_detail_")
            print(f"📊 Detailed results saved to: {detail_path}")
        
        # Show final summary
        if crawler.results:
            print(f"\n📈 Final Results Summary:")
            print(f"   - Total contact URLs found: {len(crawler.results)}")
            
            domains_with_contacts = set(r['domain'] for r in crawler.results)
            print(f"   - Domains with contact forms: {len(domains_with_contacts)}")
            
            if result_limit and len(crawler.results) >= result_limit:
                print(f"   - 🎯 Result limit of {result_limit} reached")
            
            # Show domains with most contact URLs
            domain_counts = {}
            for result in crawler.results:
                domain = result['domain']
                domain_counts[domain] = domain_counts.get(domain, 0) + 1
            
            if domain_counts:
                print(f"\n🔝 Top domains by contact URL count:")
                sorted_domains = sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)
                for domain, count in sorted_domains[:3]:  # Show top 3
                    print(f"   - {domain}: {count} URLs")
        else:
            print("❌ No contact URLs found")
        
        return True
        
    except KeyboardInterrupt:
        print(f"\n⏹️ Crawling interrupted by user")
        return False
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Parse arguments and run test"""
    parser = argparse.ArgumentParser(
        description="Test Contact Form Crawler with result limit option",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_contact_crawler.py                                    # Uses live_test_sample.csv
  python test_contact_crawler.py --csv my_domains.csv              # Uses custom CSV file
  python test_contact_crawler.py --csv my_domains.csv --limit 5    # Custom CSV with 5 result limit
  python test_contact_crawler.py --limit 10 --output-dir test_results
  python test_contact_crawler.py --csv large_list.csv --limit 20 --detail    # With detailed output
        """
    )
    
    parser.add_argument(
        "--csv",
        type=str,
        default="live_test_sample.csv",
        help="CSV file containing domains to crawl (default: live_test_sample.csv)"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop crawling after finding this many contact URLs (default: no limit)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="contact_results",
        help="Output directory for results (default: contact_results)"
    )
    
    parser.add_argument(
        "--detail",
        action="store_true",
        help="Save detailed CSV with all fields (domain, link text, confidence, etc.)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.limit is not None and args.limit <= 0:
        print("❌ Error: --limit must be a positive integer")
        sys.exit(1)
    
    # Run test
    success = asyncio.run(test_crawler(
        csv_file=args.csv,
        result_limit=args.limit,
        output_dir=args.output_dir,
        save_detail=args.detail
    ))
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()