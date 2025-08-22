# main_update_script.py
"""Main script to run the gold futures data update"""

import argparse
import os
import sys
from pathlib import Path

from gc_data_updater import GCDataUpdater
from gc_data_validator import GCDataValidator


def main():
    # Get the project root directory (where this script is located)
    project_root = Path("~/fks").expanduser()
    default_csv_path = project_root / "data" / "raw_gc_data.csv"

    parser = argparse.ArgumentParser(description="Update gold futures data")
    parser.add_argument(
        "--csv-path",
        default=str(default_csv_path),
        help="Path to CSV file (default: ./data/raw_gc_data.csv)",
    )
    parser.add_argument("--news-api-key", help="News API key for fetching headlines")
    parser.add_argument(
        "--no-headlines", action="store_true", help="Skip fetching headlines"
    )
    parser.add_argument(
        "--validate-only", action="store_true", help="Only validate existing data"
    )

    args = parser.parse_args()

    # Convert to Path object for better path handling
    csv_path = Path(args.csv_path)

    # Check if data directory exists, create if not
    if not csv_path.parent.exists():
        print(f"Creating directory: {csv_path.parent}")
        csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if file exists (for validation mode)
    if args.validate_only and not csv_path.exists():
        print(f"Error: CSV file not found at {csv_path}")
        print(f"Cannot validate non-existent file.")
        sys.exit(1)

    # Get API key from environment if not provided
    api_key = args.news_api_key or os.getenv("NEWS_API_KEY")

    # Convert Path back to string for the modules
    csv_path_str = str(csv_path)

    if args.validate_only:
        print(f"Validating data file: {csv_path}")
        validator = GCDataValidator(csv_path_str)
        results = validator.validate_all()

        print("\n" + "=" * 50)
        print("VALIDATION RESULTS")
        print("=" * 50)
        print(f"File: {csv_path}")
        print(f"Total rows: {results['total_rows']}")
        print(
            f"Date range: {results['date_range']['start']} to {results['date_range']['end']}"
        )

        # Report issues
        if results["date_order_issues"]:
            print(f"\n⚠️  Date order issues: {len(results['date_order_issues'])}")
            for i, (idx, prev, curr) in enumerate(results["date_order_issues"][:5]):
                print(f"   {i+1}. Row {idx}: {prev} -> {curr}")
            if len(results["date_order_issues"]) > 5:
                print(f"   ... and {len(results['date_order_issues']) - 5} more")

        if results["missing_dates"]:
            print(f"\n⚠️  Missing dates: {len(results['missing_dates'])}")
            for i, date in enumerate(results["missing_dates"][:5]):
                print(f"   {i+1}. {date}")
            if len(results["missing_dates"]) > 5:
                print(f"   ... and {len(results['missing_dates']) - 5} more")

        if results["duplicate_dates"]:
            print(f"\n⚠️  Duplicate dates: {len(results['duplicate_dates'])}")
            for i, date in enumerate(results["duplicate_dates"][:5]):
                print(f"   {i+1}. {date}")

        # Price issues summary
        price_issues = results.get("price_issues", {})
        total_price_issues = sum(len(v) for v in price_issues.values() if v)
        if total_price_issues > 0:
            print(f"\n⚠️  Price integrity issues:")
            for issue_type, indices in price_issues.items():
                if indices:
                    print(f"   - {issue_type}: {len(indices)} rows")

        if not any(
            [
                results["date_order_issues"],
                results["missing_dates"],
                results["duplicate_dates"],
                total_price_issues,
            ]
        ):
            print("\n✅ No data quality issues found!")

    else:
        print(f"Updating gold futures data: {csv_path}")
        print(f"Headlines: {'Enabled' if not args.no_headlines else 'Disabled'}")
        if api_key:
            print(f"News API Key: {'Set' if api_key else 'Not set'}")

        updater = GCDataUpdater(csv_path_str, api_key)
        updater.update_data(fetch_headlines=not args.no_headlines)

        print("\n✅ Update complete!")


if __name__ == "__main__":
    main()
