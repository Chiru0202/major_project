import io
import unittest

from app import app
from recommendation.advisor import generate_advice



class DashboardTestCase(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_home_page_renders(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn('Energy Dashboard', response.get_data(as_text=True))

    def test_house_and_period_filters(self):
        response = self.client.get('/?house_id=House_1&period=week')
        self.assertEqual(response.status_code, 200)
        self.assertIn('House 1', response.get_data(as_text=True))
        self.assertIn('Week Forecast', response.get_data(as_text=True))

    def test_recommendation_uses_an_alternative_when_focus_is_highest(self):
        advice = generate_advice(
            {"percentage": {"AC": 60, "Fans": 25, "TV": 15}, "peak_month": None},
            user_preference="AC",
        )
        self.assertTrue(any("If you increase AC, offset it through Fans" in item for item in advice))
        self.assertFalse(any("reduce AC" in item.lower() for item in advice))

    def test_recommendation_uses_highest_load_when_focus_is_not_highest(self):
        advice = generate_advice(
            {"percentage": {"AC": 60, "Fans": 25, "TV": 15}, "peak_month": None},
            user_preference="TV",
        )
        self.assertTrue(any("If you increase TV, offset it through AC" in item for item in advice))

    def test_recommendations_use_appliance_specific_actions(self):
        ac_advice = generate_advice(
            {"percentage": {"AC": 60, "Fans": 25, "TV": 15}, "peak_month": None},
            user_preference="AC",
        )
        washing_advice = generate_advice(
            {"percentage": {"Washing Machine": 60, "Fans": 25, "TV": 15}, "peak_month": None},
            user_preference="Washing Machine",
        )
        self.assertTrue(any("thermostat" in item for item in ac_advice))
        self.assertTrue(any("full loads" in item for item in washing_advice))
        self.assertNotEqual(ac_advice[0], washing_advice[0])

    def test_custom_house_upload(self):
        csv_data = b'timestamp,house_id,ac,fridge,lights,fans,washing_machine,tv\n2023-01-01 00:00:00,House_21,0.10,0.20,0.30,0.25,0.15,0.18\n2023-01-01 01:00:00,House_21,0.12,0.18,0.28,0.22,0.18,0.16\n'
        response = self.client.post('/', data={'new_house_name': 'House_21', 'house_file': (io.BytesIO(csv_data), 'house_21.csv')}, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 200)
        self.assertIn('House 21', response.get_data(as_text=True))

    def test_custom_house_upload_handles_missing_appliance_columns(self):
        csv_data = b'timestamp,house_id,ac,fridge,lights,fan\n2023-01-01 00:00:00,House_22,0.10,0.20,0.30,0.25\n2023-01-01 01:00:00,House_22,0.12,0.18,0.28,0.22\n'
        response = self.client.post('/', data={'new_house_name': 'House_22', 'house_file': (io.BytesIO(csv_data), 'house_22.csv')}, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 200)
        self.assertIn('House 22', response.get_data(as_text=True))

    def test_custom_house_name_overrides_csv_house_id(self):
        csv_data = b'timestamp,house_id,ac,fridge,lights,fans,washing_machine,tv\n2023-01-01 00:00:00,House_2,0.10,0.20,0.30,0.25,0.15,0.18\n'
        response = self.client.post('/', data={'new_house_name': 'House_23', 'house_file': (io.BytesIO(csv_data), 'house_23.csv')}, content_type='multipart/form-data')
        body = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn('House 23 Energy Insights', body)

        follow_up = self.client.get('/?house_id=House_23&period=day')
        self.assertEqual(follow_up.status_code, 200)
        self.assertIn('House 23', follow_up.get_data(as_text=True))


if __name__ == '__main__':
    unittest.main()
