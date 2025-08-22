"""Module for fetching gold futures EOD data"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import requests
import yfinance as yf


class GCDataFetcher:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.ticker_symbol = "GC=F"  # Gold futures symbol for yfinance

    def fetch_eod_data(self, start_date: str, end_date: str = "") -> pd.DataFrame:
        """Fetch EOD data for gold futures"""
        if end_date is None or end_date == "":
            end_date = datetime.now().strftime("%Y-%m-%d")

        self.logger.info(f"Fetching GC futures data from {start_date} to {end_date}")

        try:
            # Using yfinance as a reliable source
            ticker = yf.Ticker(self.ticker_symbol)
            df = ticker.history(start=start_date, end=end_date)

            # Rename columns to match our format
            df = df.rename(
                columns={
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                }
            )

            # Add metadata columns
            df["currency"] = "usd"
            df["unit"] = "ounce"
            df["headlines"] = ""  # Will be filled by headlines fetcher

            # Reset index to get date as column
            df.reset_index(inplace=True)
            df.rename(columns={"Date": "timestamp"}, inplace=True)

            # Format timestamp
            df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d")

            return df[
                [
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "currency",
                    "unit",
                    "headlines",
                ]
            ]

        except Exception as e:
            self.logger.error(f"Error fetching data: {str(e)}")
            raise

    def fetch_latest_data(self, last_date: str) -> pd.DataFrame:
        """Fetch data since the last recorded date"""
        start_date = (pd.to_datetime(last_date) + timedelta(days=1)).strftime(
            "%Y-%m-%d"
        )
        end_date = datetime.now().strftime("%Y-%m-%d")
        return self.fetch_eod_data(start_date, end_date)
