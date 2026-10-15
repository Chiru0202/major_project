"""Tests for the forecasting and analysis modules."""

import tempfile
import unittest
from pathlib import Path
from datetime import date, timedelta
from unittest.mock import patch

import pandas as pd

import app as dashboard
from analysis.analyzer import analyze_usage
from analysis.preprocessing import load_and_preprocess
from models.forecast import (
    EnergyForecaster,
    generate_forecast,
    generate_next_month_forecast,
    infer_frequency,
    forecast_summary,
    ForecastHistory,
)
from recommendation.advisor import generate_balancing_recommendation


class ForecastingTests(unittest.TestCase):
    """Test cases for forecasting functionality."""

    def test_energy_forecaster_basic(self):
        """Test basic forecasting functionality."""
        data = pd.Series([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
        forecaster = EnergyForecaster(data)
        forecaster.fit()
        pred, lower, upper = forecaster.predict(5)

        self.assertEqual(len(pred), 5)
        self.assertEqual(len(lower), 5)
        self.assertEqual(len(upper), 5)
        self.assertTrue(all(p >= 0 for p in pred))

    def test_forecast_summary(self):
        """Test forecast summary calculation."""
        rows = [
            {"consumption": 10, "status": "Predicted"},
            {"consumption": 12, "status": "Forecast"},
            {"consumption": 11, "status": "Forecast"},
        ]
        summary = forecast_summary(rows)

        self.assertEqual(summary["count"], 3)
        self.assertEqual(summary["total"], 33)
        self.assertEqual(summary["average"], 11)

    def test_infer_frequency_daily(self):
        """Test frequency detection for daily data."""
        dates = pd.date_range("2026-10-01", periods=10, freq="D")
        freq = infer_frequency(dates)

        self.assertEqual(freq["kind"], "daily")
        self.assertEqual(freq["label"], "Daily")

    def test_infer_frequency_hourly(self):
        """Test frequency detection for hourly data."""
        dates = pd.date_range("2026-10-01", periods=48, freq="h")  # lowercase 'h' for pandas 2.x
        freq = infer_frequency(dates)

        self.assertEqual(freq["kind"], "hourly")

    def test_infer_frequency_monthly(self):
        """Test frequency detection for monthly data."""
        dates = pd.to_datetime(["2026-01-01", "2026-02-01", "2026-03-01"])
        freq = infer_frequency(dates)

        self.assertEqual(freq["kind"], "monthly")

    def test_generate_forecast_returns_list(self):
        """Test that generate_forecast returns a list of dicts."""
        df = pd.DataFrame({
            "date": pd.date_range("2026-10-01", periods=30, freq="D"),
            "total_usage": [10 + i * 0.1 for i in range(30)],
        })
        forecast = generate_forecast(df, period="week")

        self.assertIsInstance(forecast, list)
        if forecast:
            self.assertIn("date", forecast[0])
            self.assertIn("consumption", forecast[0])

    def test_preprocessing_detects_date_column(self):
        """Test that preprocessing detects various date column names."""
        source = "Timestamp,Consumption\n2026-10-01,100\n2026-10-02,110\n"
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "test.csv"
            path.write_text(source)
            df = load_and_preprocess(path)

        self.assertIn("date", df.columns)
        self.assertEqual(len(df), 2)

    def test_preprocessing_calculates_total(self):
        """Test that preprocessing calculates total from appliances."""
        source = "date,fan,lights\n2026-10-01,1.0,2.0\n2026-10-02,1.5,2.5\n"
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "test.csv"
            path.write_text(source)
            df = load_and_preprocess(path)

        self.assertIn("total_usage", df.columns)
        self.assertEqual(df.loc[0, "total_usage"], 3.0)

    def test_analyze_usage_returns_complete_dict(self):
        """Test that analyze_usage returns all expected keys."""
        df = pd.DataFrame({
            "date": pd.date_range("2026-10-01", periods=10, freq="D"),
            "total_usage": [10 + i for i in range(10)],
            "fan": [5 + i * 0.5 for i in range(10)],
        })
        analysis = analyze_usage(df)

        self.assertIn("average_consumption", analysis)
        self.assertIn("trend", analysis)
        self.assertIn("appliance_rows", analysis)
        self.assertIn("percentage", analysis)

    def test_analyze_usage_detects_trend(self):
        """Test that analyze_usage detects increasing trend."""
        df = pd.DataFrame({
            "date": pd.date_range("2026-10-01", periods=20, freq="D"),
            "total_usage": [10 + i * 2 for i in range(20)],
        })
        analysis = analyze_usage(df)

        self.assertEqual(analysis["trend"], "Increasing")

    def test_balancing_recommendation(self):
        """Test balancing recommendation generation."""
        df = pd.DataFrame({
            "date": pd.date_range("2026-10-01", periods=10, freq="D"),
            "ac": [5.0] * 10,
            "lights": [2.0] * 10,
            "fan": [1.0] * 10,
            "total_usage": [8.0] * 10,
        })
        rec = generate_balancing_recommendation(df, "Air Conditioner", additional_kwh=1.0)

        self.assertIn("1.0", rec)
        self.assertIsInstance(rec, str)

    def test_3month_forecast(self):
        """Test 3-month forecast period."""
        df = pd.DataFrame({
            "date": pd.date_range("2026-10-01", periods=30, freq="D"),
            "total_usage": [10 + i * 0.1 for i in range(30)],
        })
        forecast = generate_forecast(df, period="3month")

        self.assertIsInstance(forecast, list)
        self.assertGreater(len(forecast), 0)

    def test_forecast_provisional_status(self):
        """Test that long forecasts include provisional status."""
        df = pd.DataFrame({
            "date": pd.date_range("2026-10-01", periods=100, freq="D"),
            "total_usage": [10 + i * 0.1 for i in range(100)],
        })
        forecast = generate_forecast(df, period="3month", recursive=True)

        statuses = {r["status"] for r in forecast}
        self.assertIn("Predicted", statuses)
        # For long forecasts, should have provisional
        if len(forecast) > 30:
            self.assertIn("Provisional", statuses)

    def test_forecast_history_save_and_retrieve(self):
        """Test forecast history persistence."""
        with tempfile.TemporaryDirectory() as d:
            house_path = Path(d) / "test_house"
            house_path.mkdir()

            history = ForecastHistory(house_path)

            forecast_rows = [
                {"date": "2026-11-01 00:00:00", "consumption": 10.5, "lower": 9.0, "upper": 12.0},
                {"date": "2026-11-02 00:00:00", "consumption": 11.0, "lower": 9.5, "upper": 12.5},
            ]

            saved = history.save_forecast(forecast_rows, model_order=(1, 1, 1), period="week")
            self.assertEqual(saved, 2)

            records = history.get_history()
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]["status"], "Provisional")

    def test_forecast_history_confirm_with_actual(self):
        """Test updating forecasts with actual data."""
        with tempfile.TemporaryDirectory() as d:
            house_path = Path(d) / "test_house"
            house_path.mkdir()

            history = ForecastHistory(house_path)

            forecast_rows = [
                {"date": "2026-11-01 00:00:00", "consumption": 10.5},
            ]
            history.save_forecast(forecast_rows)

            actual_df = pd.DataFrame({
                "date": pd.to_datetime(["2026-11-01 00:00:00"]),
                "total_usage": [10.8],
            })

            updated = history.confirm_with_actual(actual_df)
            self.assertEqual(updated, 1)

            records = history.get_history()
            self.assertEqual(records[0]["status"], "Confirmed")
            self.assertEqual(records[0]["actual"], 10.8)


class DashboardIntegrationTests(unittest.TestCase):
    """Integration tests for the dashboard."""

    def test_dashboard_renders_with_data(self):
        """Test that dashboard renders successfully with real data."""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            data_dir = root / "data"
            house_dir = data_dir / "houses" / "test"
            house_dir.mkdir(parents=True)

            # Create test data
            df = pd.DataFrame({
                "date": pd.date_range("2026-10-01", periods=10, freq="D"),
                "fan": [1.0] * 10,
                "total_usage": [2.0] * 10,
            })
            dataset_path = house_dir / "energy.csv"
            df.to_csv(dataset_path, index=False)

            with (
                patch.object(dashboard, "DATA_DIR", data_dir),
                patch.object(dashboard, "HOUSE_DIR", data_dir / "houses"),
                patch.object(dashboard, "REGISTRY_PATH", data_dir / "houses.json"),
                dashboard.app.test_client() as client,
            ):
                dashboard.save_registry([{
                    "id": "test",
                    "name": "Test House",
                    "dataset": str(dataset_path),
                }])

                response = client.get("/?house_id=test")

                self.assertEqual(response.status_code, 200)
                body = response.get_data(as_text=True)
                self.assertIn("Test House", body)
                self.assertIn("Energy Dashboard", body)


if __name__ == "__main__":
    unittest.main()
