"""Pruebas de correspondencia de estaciones contra el catálogo."""

import pandas as pd
import pytest

from mi_bici.data_quality import validate_station_coverage


@pytest.mark.parametrize(
    "origins,destinations,catalog,expected_counts",
    [
        pytest.param(
            ["A", "B"],
            ["B", "A"],
            {"A", "B"},
            (0, 0, 0, 0),
            id="complete-coverage",
        ),
        pytest.param(
            [None, "", "   ", "A"],
            ["A", None, "", "\t"],
            {"A"},
            (3, 3, 0, 0),
            id="missing-identifiers",
        ),
        pytest.param(
            ["X", "X", "A"],
            ["A", "Y", "Y"],
            {"A"},
            (0, 0, 2, 2),
            id="unknown-stations-counted-per-trip",
        ),
        pytest.param(
            ["A"],
            ["A"],
            {"A", "B"},
            (0, 0, 0, 0),
            id="catalog-station-without-trips",
        ),
        pytest.param(
            [None, "X", "A"],
            ["Y", "", "A"],
            {"A"},
            (1, 1, 1, 1),
            id="missing-and-unknown-are-distinct",
        ),
        pytest.param(
            [],
            [],
            {"A"},
            (0, 0, 0, 0),
            id="empty-trips",
        ),
    ],
)
def test_validate_station_coverage(tmp_path, origins, destinations, catalog, expected_counts):
    trips = pd.DataFrame(
        {
            "origin_station_id": pd.Series(origins, dtype="string"),
            "destination_station_id": pd.Series(destinations, dtype="string"),
        }
    )
    path = tmp_path / "viajes_2026_06.parquet"
    trips.to_parquet(path, index=False)

    result = validate_station_coverage(path, catalog)

    missing_origin, missing_destination, unknown_origins, unknown_destinations = expected_counts
    assert result == {
        "period": "2026-06",
        "rows": len(trips),
        "missing_origin": missing_origin,
        "missing_destination": missing_destination,
        "unknown_origins": unknown_origins,
        "unknown_destinations": unknown_destinations,
    }
