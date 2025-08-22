# gc_data_updater.py
"""Main module for updating gold futures data"""

import logging
import os
from datetime import datetime
from typing import Optional

import pandas as pd
from gc_data_fetcher import GCDataFetcher
from gc_data_validator import GCDataValidator
from gc_headlines_fetcher import GCHeadlinesFetcher


class GCDataUpdater:
    def __init__(self, csv_path: str, news_api_key: Optional[str] = None):
        self.csv_path = csv_path
        self.validator = GCDataValidator(csv_path)
        self.fetcher = GCDataFetcher()
        self.headlines_fetcher = GCHeadlinesFetcher(
            news_api_key if news_api_key is not None else ""
        )

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)

    def update_data(self, fetch_headlines: bool = True):
        """Main update process"""
        self.logger.info("Starting gold futures data update process")

        # Load existing data
        try:
            df_existing = pd.read_csv(
                self.csv_path, delimiter=";", parse_dates=["timestamp"]
            )
            last_date = df_existing["timestamp"].max()
            self.logger.info(f"Last date in existing data: {last_date}")
        except FileNotFoundError:
            self.logger.warning(f"File {self.csv_path} not found. Creating new file.")
            df_existing = pd.DataFrame()
            last_date = "2000-01-01"

        # Fetch new data
        df_new = self.fetcher.fetch_latest_data(
            last_date.strftime("%Y-%m-%d")
            if isinstance(last_date, pd.Timestamp)
            else last_date
        )

        if df_new.empty:
            self.logger.info("No new data to update")
            return

        self.logger.info(f"Fetched {len(df_new)} new records")

        # Fetch headlines if requested
        if fetch_headlines and not df_new.empty:
            self.logger.info("Fetching headlines for new dates")
            headlines_dict = self.headlines_fetcher.update_headlines_batch(
                df_new["timestamp"].tolist()
            )
            df_new["headlines"] = df_new["timestamp"].map(headlines_dict).fillna("")

        # Combine data
        if not df_existing.empty:
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_combined = df_new

        # Sort by date
        df_combined = df_combined.sort_values("timestamp")

        # Remove duplicates
        df_combined = df_combined.drop_duplicates(subset=["timestamp"], keep="last")

        # Save backup
        backup_path = (
            f"{self.csv_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        if os.path.exists(self.csv_path):
            os.rename(self.csv_path, backup_path)
            self.logger.info(f"Created backup: {backup_path}")

        # Save updated data
        df_combined.to_csv(self.csv_path, sep=";", index=False)
        self.logger.info(f"Updated data saved to {self.csv_path}")

        # Validate updated data
        validation_results = self.validator.validate_all()
        self._log_validation_results(validation_results)

    def _log_validation_results(self, results: dict):
        """Log validation results"""
        self.logger.info(f"Total rows: {results['total_rows']}")
        self.logger.info(
            f"Date range: {results['date_range']['start']} to {results['date_range']['end']}"
        )

        if results["date_order_issues"]:
            self.logger.warning(
                f"Found {len(results['date_order_issues'])} date order issues"
            )

        if results["missing_dates"]:
            self.logger.warning(f"Found {len(results['missing_dates'])} missing dates")

        price_issues = results["price_issues"]
        for issue_type, indices in price_issues.items():
            if indices:
                self.logger.warning(f"Found {len(indices)} {issue_type} issues")
