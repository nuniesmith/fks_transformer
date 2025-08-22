import os
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from .main_update_script import main


class TestMainUpdateScript:

    @patch("sys.argv", ["main_update_script.py"])
    @patch("builtins.print")
    @patch("main_update_script.GCDataUpdater")
    def test_main_with_default_arguments(self, mock_updater_class, mock_print):
        """Test main with default arguments"""
        mock_updater = Mock()
        mock_updater_class.return_value = mock_updater

        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.mkdir"):
                main()

        # Verify updater was created with default path
        expected_path = str(Path("~/fks/data/raw_gc_data.csv").expanduser())
        mock_updater_class.assert_called_once_with(expected_path, None)
        mock_updater.update_data.assert_called_once_with(fetch_headlines=True)

    @patch(
        "sys.argv",
        [
            "main_update_script.py",
            "--csv-path",
            "/custom/path.csv",
            "--news-api-key",
            "test-key",
        ],
    )
    @patch("builtins.print")
    @patch("main_update_script.GCDataUpdater")
    def test_main_with_custom_arguments(self, mock_updater_class, mock_print):
        """Test main with custom CSV path and API key"""
        mock_updater = Mock()
        mock_updater_class.return_value = mock_updater

        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.mkdir"):
                main()

        mock_updater_class.assert_called_once_with("/custom/path.csv", "test-key")
        mock_updater.update_data.assert_called_once_with(fetch_headlines=True)

    @patch("sys.argv", ["main_update_script.py", "--no-headlines"])
    @patch("builtins.print")
    @patch("main_update_script.GCDataUpdater")
    def test_main_with_no_headlines(self, mock_updater_class, mock_print):
        """Test main with headlines disabled"""
        mock_updater = Mock()
        mock_updater_class.return_value = mock_updater

        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.mkdir"):
                main()

        mock_updater.update_data.assert_called_once_with(fetch_headlines=False)

    @patch("os.getenv", return_value="env-api-key")
    @patch("sys.argv", ["main_update_script.py"])
    @patch("builtins.print")
    @patch("main_update_script.GCDataUpdater")
    def test_main_with_env_api_key(self, mock_updater_class, mock_print, mock_getenv):
        """Test main uses environment variable for API key"""
        mock_updater = Mock()
        mock_updater_class.return_value = mock_updater

        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.mkdir"):
                main()

        expected_path = str(Path("~/fks/data/raw_gc_data.csv").expanduser())
        mock_updater_class.assert_called_once_with(expected_path, "env-api-key")
        mock_getenv.assert_called_once_with("NEWS_API_KEY")

    @patch("sys.argv", ["main_update_script.py", "--validate-only"])
    @patch("builtins.print")
    @patch("main_update_script.GCDataValidator")
    def test_validate_only_mode_success(self, mock_validator_class, mock_print):
        """Test validation-only mode with successful validation"""
        mock_validator = Mock()
        mock_validator_class.return_value = mock_validator

        # Mock validation results with no issues
        mock_validator.validate_all.return_value = {
            "total_rows": 100,
            "date_range": {"start": "2024-01-01", "end": "2024-12-31"},
            "date_order_issues": [],
            "missing_dates": [],
            "duplicate_dates": [],
            "price_issues": {},
        }

        with patch("pathlib.Path.exists", return_value=True):
            main()

        expected_path = str(Path("~/fks/data/raw_gc_data.csv").expanduser())
        mock_validator_class.assert_called_once_with(expected_path)
        mock_validator.validate_all.assert_called_once()

    @patch("sys.argv", ["main_update_script.py", "--validate-only"])
    @patch("builtins.print")
    @patch("main_update_script.GCDataValidator")
    def test_validate_only_mode_with_issues(self, mock_validator_class, mock_print):
        """Test validation-only mode with validation issues"""
        mock_validator = Mock()
        mock_validator_class.return_value = mock_validator

        # Mock validation results with issues
        mock_validator.validate_all.return_value = {
            "total_rows": 100,
            "date_range": {"start": "2024-01-01", "end": "2024-12-31"},
            "date_order_issues": [(10, "2024-01-10", "2024-01-05")],
            "missing_dates": ["2024-01-15"],
            "duplicate_dates": ["2024-01-20"],
            "price_issues": {"negative_prices": [25], "zero_volume": [30]},
        }

        with patch("pathlib.Path.exists", return_value=True):
            main()

        mock_validator.validate_all.assert_called_once()
        # Verify print was called multiple times for different issue types
        assert mock_print.call_count > 10

    @patch("sys.argv", ["main_update_script.py", "--validate-only"])
    @patch("sys.exit")
    @patch("builtins.print")
    def test_validate_only_file_not_exists(self, mock_print, mock_exit):
        """Test validation-only mode when file doesn't exist"""
        with patch("pathlib.Path.exists", return_value=False):
            main()

        mock_exit.assert_called_once_with(1)
        mock_print.assert_any_call(
            "Error: CSV file not found at ~/fks/data/raw_gc_data.csv"
        )

    @patch("sys.argv", ["main_update_script.py"])
    @patch("builtins.print")
    @patch("main_update_script.GCDataUpdater")
    def test_directory_creation(self, mock_updater_class, mock_print):
        """Test directory creation when parent doesn't exist"""
        mock_updater = Mock()
        mock_updater_class.return_value = mock_updater

        with patch("pathlib.Path.exists", return_value=False):
            with patch("pathlib.Path.mkdir") as mock_mkdir:
                main()

        # Verify directory creation was attempted
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

    @patch("sys.argv", ["main_update_script.py", "--validate-only"])
    @patch("builtins.print")
    @patch("main_update_script.GCDataValidator")
    def test_validate_with_many_issues(self, mock_validator_class, mock_print):
        """Test validation output with many issues (tests truncation)"""
        mock_validator = Mock()
        mock_validator_class.return_value = mock_validator

        # Create more than 5 issues to test truncation
        mock_validator.validate_all.return_value = {
            "total_rows": 100,
            "date_range": {"start": "2024-01-01", "end": "2024-12-31"},
            "date_order_issues": [
                (i, f"2024-01-{i:02d}", f"2024-01-{i-1:02d}") for i in range(1, 8)
            ],
            "missing_dates": [f"2024-02-{i:02d}" for i in range(1, 8)],
            "duplicate_dates": [f"2024-03-{i:02d}" for i in range(1, 8)],
            "price_issues": {"negative_prices": list(range(10))},
        }

        with patch("pathlib.Path.exists", return_value=True):
            main()

        # Verify truncation messages are printed
        print_calls = [str(call) for call in mock_print.call_args_list]
        truncation_messages = [
            call for call in print_calls if "... and" in call and "more" in call
        ]
        assert (
            len(truncation_messages) >= 2
        )  # Should have truncation for date_order_issues and missing_dates

    @patch(
        "sys.argv",
        [
            "main_update_script.py",
            "--csv-path",
            "/test/path.csv",
            "--news-api-key",
            "override-key",
        ],
    )
    @patch("os.getenv", return_value="env-key")
    @patch("builtins.print")
    @patch("main_update_script.GCDataUpdater")
    def test_api_key_argument_overrides_env(
        self, mock_updater_class, mock_print, mock_getenv
    ):
        """Test that command line API key overrides environment variable"""
        mock_updater = Mock()
        mock_updater_class.return_value = mock_updater

        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.mkdir"):
                main()

        # Should use the command line argument, not environment variable
        mock_updater_class.assert_called_once_with("/test/path.csv", "override-key")
        # Environment variable should not be called when argument is provided
        mock_getenv.assert_not_called()

    @patch("sys.argv", ["main_update_script.py"])
    @patch("os.getenv", return_value=None)
    @patch("builtins.print")
    @patch("main_update_script.GCDataUpdater")
    def test_no_api_key_provided(self, mock_updater_class, mock_print, mock_getenv):
        """Test when no API key is provided via argument or environment"""
        mock_updater = Mock()
        mock_updater_class.return_value = mock_updater

        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.mkdir"):
                main()

        expected_path = str(Path("~/fks/data/raw_gc_data.csv").expanduser())
        mock_updater_class.assert_called_once_with(expected_path, None)
        mock_getenv.assert_called_once_with("NEWS_API_KEY")
