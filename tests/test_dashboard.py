"""Tests for the Energy Dashboard application."""

import io
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import app as dashboard
from recommendation.advisor import generate_advice

app = dashboard.app


class DashboardTestCase(unittest.TestCase):
    """Test cases for the dashboard routes."""

    def setUp(self):
        """Set up test fixtures."""
        self.temporary_directory = tempfile.TemporaryDirectory()
        data_dir = Path(self.temporary_directory.name) / "data"
        house_dir = data_dir / "houses"

        self.patchers = [
            patch.object(dashboard, "DATA_DIR", data_dir),
            patch.object(dashboard, "HOUSE_DIR", house_dir),
            patch.object(dashboard, "REGISTRY_PATH", data_dir / "houses.json"),
        ]
        for patcher in self.patchers:
            patcher.start()

        # Create test house
        house_id = "house_1"
        house_path = house_dir / house_id
        house_path.mkdir(parents=True)

        source = pd.read_csv(dashboard.DEFAULT_DATASET)
        source = source[source["house_id"].astype(str) == "House_1"]
        dataset_path = house_path / "energy.csv"
        source.to_csv(dataset_path, index=False)

        dashboard.save_registry([{
            "id": house_id,
            "name": "House 1",
            "dataset": str(dataset_path),
        }])

        self.client = app.test_client()

    def tearDown(self):
        """Clean up test fixtures."""
        for patcher in reversed(self.patchers):
            patcher.stop()
        self.temporary_directory.cleanup()

    def test_home_page_renders(self):
        """Test that home page renders successfully."""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn('Energy Dashboard', response.get_data(as_text=True))

    def test_usage_breakdown_is_in_ascending_order(self):
        """Test that usage breakdown is sorted ascending."""
        response = self.client.get('/')
        body = response.get_data(as_text=True)
        # Check that usage distribution section exists
        self.assertIn('Usage Distribution', body)
        # Find percentages in the usage breakdown
        percentages = [float(v) for v in re.findall(r':\s*<strong>([\d.]+)%</strong>', body)]
        if percentages:
            self.assertEqual(percentages, sorted(percentages))

    def test_house_and_period_filters(self):
        """Test house and period filtering."""
        response = self.client.get('/?house_id=House_1&period=week')
        self.assertEqual(response.status_code, 200)
        body = response.get_data(as_text=True)
        self.assertIn('House 1', body)

    def test_recommendation_includes_offset_suggestion(self):
        """Test that recommendations include offset suggestions."""
        advice = generate_advice(
            {"percentage": {"Air Conditioner": 60, "Fan": 25, "Television": 15}, "peak_month": None},
            user_preference="Air Conditioner",
        )
        self.assertTrue(any("offset" in item.lower() or "reduce" in item.lower() for item in advice))

    def test_recommendations_are_generated(self):
        """Test that recommendations are generated."""
        advice = generate_advice(
            {"percentage": {"Air Conditioner": 60, "Fan": 25, "Television": 15}, "peak_month": None},
        )
        self.assertGreater(len(advice), 0)

    def test_custom_house_upload(self):
        """Test uploading a custom house dataset."""
        csv_data = b'timestamp,house_id,ac,fridge,lights,fans,washing_machine,tv\n2023-01-01 00:00:00,House_21,0.10,0.20,0.30,0.25,0.15,0.18\n2023-01-01 01:00:00,House_21,0.12,0.18,0.28,0.22,0.18,0.16\n'
        response = self.client.post(
            '/',
            data={'new_house_name': 'House_21', 'house_file': (io.BytesIO(csv_data), 'house_21.csv')},
            content_type='multipart/form-data'
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn('House 21', response.get_data(as_text=True))

    def test_invalid_dataset_upload_shows_error(self):
        """Test that invalid dataset shows friendly error."""
        response = self.client.post(
            '/',
            data={
                'new_house_name': 'Invalid House',
                'house_file': (io.BytesIO(b'invalid,data\nno-date,1.0\n'), 'invalid.csv'),
            },
            content_type='multipart/form-data',
        )
        self.assertEqual(response.status_code, 200)
        body = response.get_data(as_text=True)
        self.assertTrue('date' in body.lower() or 'error' in body.lower())


if __name__ == '__main__':
    unittest.main()
