# gc_data_validator.py
"""Module for validating gold futures data integrity"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


class GCDataValidator:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.logger = logging.getLogger(__name__)

    def load_data(self) -> pd.DataFrame:
        """Load CSV data with proper data types"""
        df = pd.read_csv(self.csv_path, delimiter=";", parse_dates=["timestamp"])
        return df

    def check_date_order(self, df: pd.DataFrame) -> List[Tuple[int, str, str]]:
        """Check if dates are in ascending order"""
        issues = []
        for i in range(1, len(df)):
            if df.iloc[i]["timestamp"] <= df.iloc[i - 1]["timestamp"]:
                issues.append(
                    (
                        i,
                        df.iloc[i - 1]["timestamp"].strftime("%Y-%m-%d"),
                        df.iloc[i]["timestamp"].strftime("%Y-%m-%d"),
                    )
                )
        return issues

    def check_missing_dates(self, df: pd.DataFrame) -> List[str]:
        """Find missing trading days (excludes weekends and major holidays)"""
        df_sorted = df.sort_values("timestamp")
        start_date = df_sorted["timestamp"].min()
        end_date = df_sorted["timestamp"].max()

        # Generate all business days
        all_business_days = pd.bdate_range(start=start_date, end=end_date)
        existing_dates = pd.to_datetime(df_sorted["timestamp"]).dt.normalize()

        # Find missing dates
        missing_dates = set(all_business_days) - set(existing_dates)

        # Filter out known holidays (basic US market holidays)
        holidays = self._get_us_market_holidays(start_date.year, end_date.year)
        missing_dates = [d for d in missing_dates if d not in holidays]

        return sorted([d.strftime("%Y-%m-%d") for d in missing_dates])

    def _get_us_market_holidays(self, start_year: int, end_year: int) -> List[datetime]:
        """Get basic US market holidays (simplified version)"""
        holidays = []
        # This is a simplified version - in production, use pandas_market_calendars
        # or a proper holiday calendar
        return holidays

    def check_price_integrity(self, df: pd.DataFrame) -> Dict[str, List[int]]:
        """Check for price anomalies"""
        issues = {
            "negative_prices": [],
            "zero_prices": [],
            "high_low_mismatch": [],
            "ohlc_mismatch": [],
            "extreme_changes": [],
        }

        for idx, row in df.iterrows():
            # Check for negative prices
            if any(row[col] < 0 for col in ["open", "high", "low", "close"]):
                issues["negative_prices"].append(idx)

            # Check for zero prices
            if any(row[col] == 0 for col in ["open", "high", "low", "close"]):
                issues["zero_prices"].append(idx)

            # Check high >= low
            if row["high"] < row["low"]:
                issues["high_low_mismatch"].append(idx)

            # Check OHLC relationships
            if not (
                row["low"] <= row["open"] <= row["high"]
                and row["low"] <= row["close"] <= row["high"]
            ):
                issues["ohlc_mismatch"].append(idx)

        # Check for extreme price changes (>10% daily move)
        df["pct_change"] = df["close"].pct_change()
        extreme_moves = df[df["pct_change"].abs() > 0.10]
        issues["extreme_changes"] = extreme_moves.index.tolist()

        return issues

    def validate_all(self) -> Dict[str, Any]:
        """Run all validation checks"""
        self.logger.info(f"Starting validation of {self.csv_path}")

        df = self.load_data()
        results = {
            "total_rows": len(df),
            "date_range": {
                "start": df["timestamp"].min().strftime("%Y-%m-%d"),
                "end": df["timestamp"].max().strftime("%Y-%m-%d"),
            },
            "date_order_issues": self.check_date_order(df),
            "missing_dates": self.check_missing_dates(df),
            "price_issues": self.check_price_integrity(df),
            "duplicate_dates": (
                df[df.duplicated("timestamp", keep=False)]["timestamp"].tolist()
            ),
        }

        return results


# config.py
"""Configuration file for data sources and API keys"""

# Data source configurations
DATA_SOURCES = {
    "primary": {"name": "yfinance", "ticker": "GC=F", "interval": "1d"},
    "backup": {
        "name": "alpha_vantage",
        "symbol": "GC",
        "api_key_env": "ALPHA_VANTAGE_API_KEY",
    },
}

# News sources
NEWS_SOURCES = {
    "newsapi": {
        "api_key_env": "NEWS_API_KEY",
        "keywords": [
            "gold",
            "commodity",
            "federal reserve",
            "inflation",
            "dollar",
            "precious metals",
        ],
    }
}

# Data quality thresholds
QUALITY_THRESHOLDS = {
    "max_daily_change_pct": 0.10,  # 10% daily move threshold
    "min_volume": 0,  # Some days may have 0 volume
    "price_precision": 2,  # Decimal places for prices
}
