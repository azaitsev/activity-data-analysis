from datetime import datetime, timezone
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from src.app import dataframe_to_apex_series, parse_fit_bytes, parse_tcx_bytes


class ActivitySpeedParsingTests(TestCase):
    def test_tcx_activity_extension_speed_is_converted_to_kmh(self) -> None:
        # TCX speed lives under a separately versioned extension namespace, which previously made
        # every parsed speed series empty even though the uploaded file contained Speed values.
        tcx_bytes = b"""<?xml version="1.0" encoding="UTF-8"?>
        <TrainingCenterDatabase xmlns="http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2"
            xmlns:ns3="http://www.garmin.com/xmlschemas/ActivityExtension/v2">
          <Activities><Activity Sport="Biking"><Lap StartTime="2026-08-03T10:00:00Z"><Track>
            <Trackpoint>
              <Time>2026-08-03T10:00:00Z</Time>
              <Extensions><ns3:TPX><ns3:Speed>10.5</ns3:Speed></ns3:TPX></Extensions>
            </Trackpoint>
          </Track></Lap></Activity></Activities>
        </TrainingCenterDatabase>"""

        dataframe = parse_tcx_bytes(tcx_bytes)
        series = dataframe_to_apex_series(dataframe, "speed_kmh", "activity.tcx")

        self.assertAlmostEqual(dataframe.iloc[0]["speed_kmh"], 37.8)
        self.assertEqual(series["data"], [[1785751200000, 37.800000000000004]])

    def test_fit_enhanced_speed_populates_speed_series(self) -> None:
        # Modern FIT records may contain enhanced_speed without legacy speed, so this regression
        # fixture proves that such files no longer produce an empty speed chart.
        record = [
            SimpleNamespace(name="timestamp", value=datetime(2026, 8, 3, 10, 0, tzinfo=timezone.utc)),
            SimpleNamespace(name="enhanced_speed", value=10.5),
        ]
        fake_fit_file = SimpleNamespace(get_messages=lambda message_type: [record])

        with patch("src.app.FitFile", return_value=fake_fit_file):
            dataframe = parse_fit_bytes(b"fit fixture")
        series = dataframe_to_apex_series(dataframe, "speed_kmh", "activity.fit")

        self.assertAlmostEqual(dataframe.iloc[0]["speed_kmh"], 37.8)
        self.assertEqual(series["data"], [[1785751200000, 37.800000000000004]])

    def test_fit_enhanced_speed_is_preferred_over_legacy_speed(self) -> None:
        # Preferring enhanced_speed avoids silently downgrading precision when a FIT record exposes
        # both the legacy and enhanced fields in either parser order.
        record = [
            SimpleNamespace(name="timestamp", value=datetime(2026, 8, 3, 10, 0, tzinfo=timezone.utc)),
            SimpleNamespace(name="enhanced_speed", value=10.5),
            SimpleNamespace(name="speed", value=10.0),
        ]
        fake_fit_file = SimpleNamespace(get_messages=lambda message_type: [record])

        with patch("src.app.FitFile", return_value=fake_fit_file):
            dataframe = parse_fit_bytes(b"fit fixture")

        self.assertAlmostEqual(dataframe.iloc[0]["speed_kmh"], 37.8)
