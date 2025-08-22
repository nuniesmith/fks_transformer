# gc_headlines_fetcher.py
"""Module for fetching financial headlines"""

import logging
import os
import time
from datetime import timedelta
from typing import Dict, List

import pandas as pd
from newsapi import NewsApiClient


class GCHeadlinesFetcher:
    def __init__(self, api_key: str = ""):
        self.logger = logging.getLogger(__name__)
        self.api_key = api_key or os.getenv("NEWS_API_KEY")

        if self.api_key:
            self.newsapi = NewsApiClient(api_key=self.api_key)
        else:
            self.logger.warning("No News API key provided. Headlines will be empty.")
            self.newsapi = None

    def fetch_headlines(self, date: str, max_headlines: int = 10) -> str:
        """Fetch financial headlines for a specific date"""
        if not self.newsapi:
            return ""

        try:
            # Convert date string to datetime
            target_date = pd.to_datetime(date)

            # NewsAPI requires dates in ISO format
            from_date = target_date.strftime("%Y-%m-%d")
            to_date = (target_date + timedelta(days=1)).strftime("%Y-%m-%d")

            # Fetch headlines
            response = self.newsapi.get_everything(
                q='gold OR commodity OR "federal reserve" OR inflation OR dollar',
                from_param=from_date,
                to=to_date,
                language="en",
                sort_by="relevancy",
                page_size=max_headlines,
            )

            if response["status"] == "ok":
                headlines = [article["title"] for article in response["articles"]]
                return " / ".join(headlines[:max_headlines])
            else:
                self.logger.error(f"Error fetching headlines: {response}")
                return ""

        except Exception as e:
            self.logger.error(f"Error fetching headlines for {date}: {str(e)}")
            return ""

    def update_headlines_batch(self, dates: List[str]) -> Dict[str, str]:
        """Fetch headlines for multiple dates"""
        headlines_dict = {}

        for date in dates:
            headlines_dict[date] = self.fetch_headlines(date)
            time.sleep(0.1)  # Rate limiting

        return headlines_dict
