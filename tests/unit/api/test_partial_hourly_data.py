"""Regression tests for consumption arriving before Fortum pricing."""

from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest

from custom_components.fortum.api.client import FortumAPIClient
from custom_components.fortum.models import (
    CostDataPoint,
    EnergyDataPoint,
    MeteringPoint,
    Price,
    TemperatureReading,
    TimeSeries,
    TimeSeriesDataPoint,
)

START = datetime(2026, 11, 1, tzinfo=UTC)


def _point(hour=0, energy=1.0, *, priced=False, cost=None):
    return TimeSeriesDataPoint(
        at_utc=START + timedelta(hours=hour),
        energy=[] if energy is None else [EnergyDataPoint(energy, "ENERGY")],
        cost=None if cost is None else [CostDataPoint(cost, cost, "COST")],
        price=Price(0.0, 0.0, 0.0, 25.0) if priced else None,
        temperature_reading=TemperatureReading(5.0),
    )


@pytest.mark.parametrize("energy", [0.0, 1.25])
def test_energy_without_price_is_not_missing(mock_hass, mock_auth_client, energy):
    client = FortumAPIClient(mock_hass, mock_auth_client)
    rows = client._build_hourly_rows_from_points(
        [_point(energy=energy)], START, START + timedelta(hours=1)
    )
    assert [row["state"] for row in rows.consumption] == [energy]
    assert rows.cost == []
    assert rows.price == []


@pytest.mark.parametrize("priced", [False, True])
def test_missing_energy_is_not_zero(mock_hass, mock_auth_client, priced):
    client = FortumAPIClient(mock_hass, mock_auth_client)
    rows = client._build_hourly_rows_from_points(
        [_point(energy=None, priced=priced)], START, START + timedelta(hours=1)
    )
    assert rows.consumption == []
    assert len(rows.price) == int(priced)
    assert len(rows.temperature) == int(priced)


@pytest.mark.parametrize("priced", [False, True])
def test_non_consumption_energy_type_is_not_zero(mock_hass, mock_auth_client, priced):
    client = FortumAPIClient(mock_hass, mock_auth_client)
    point = _point(priced=priced)
    point.energy = [EnergyDataPoint(2.0, "PRODUCTION")]
    rows = client._build_hourly_rows_from_points(
        [point], START, START + timedelta(hours=1)
    )
    assert rows.consumption == []
    assert len(rows.price) == int(priced)
    assert len(rows.temperature) == int(priced)


@pytest.mark.parametrize("price", [None, -0.5, 0.0, 1.5])
@pytest.mark.parametrize("cost", [None, -0.5, 0.0, 1.5])
async def test_pricing_only_readings(hourly_store, price, cost):
    client, ids, store, series = hourly_store
    point = _point(energy=None, cost=cost)
    point.price = None if price is None else Price(price, price, 0.0, 25.0)
    series.series = [point]
    end = START + timedelta(hours=1)
    expected_hours = int(price is not None or cost is not None)

    assert await client._record_hourly_data_stats("test", START, end) == expected_hours
    assert START not in store[ids.consumption]
    if cost is None:
        assert START not in store[ids.cost]
    else:
        assert store[ids.cost][START]["state"] == cost
        assert store[ids.cost][START]["sum"] == 50.0 + cost
    if price is None:
        assert ids.price not in store
    else:
        assert store[ids.price][START]["state"] == price
    assert (START in store.get(ids.temperature, {})) == bool(expected_hours)
    assert await client._record_hourly_data_stats("test", START, end) == 0


@pytest.fixture
def hourly_store(mock_hass, mock_auth_client):
    """Exercise importer reads/writes against a deterministic in-memory Recorder."""
    client = FortumAPIClient(mock_hass, mock_auth_client)
    ids = client._build_hourly_statistic_ids("test")
    store = {
        ids.consumption: {START - timedelta(hours=1): {"state": 4.0, "sum": 100.0}},
        ids.cost: {START - timedelta(hours=1): {"state": 2.0, "sum": 50.0}},
    }
    series = TimeSeries(
        delivery_site_category="CONSUMPTION",
        measurement_unit="kWh",
        metering_point_no="test",
        price_unit="öre/kWh",
        cost_unit="SEK",
        temperature_unit="celsius",
        series=[],
    )

    async def read_values(statistic_ids, from_date, to_date, *, value_type):
        return {
            sid: {
                hour: row[value_type]
                for hour, row in store.get(sid, {}).items()
                if from_date <= hour < to_date and value_type in row
            }
            for sid in statistic_ids
        }

    async def read_price_hours(_metering_point_no, from_date, to_date):
        return {
            hour for hour in store.get(ids.price, {}) if from_date <= hour < to_date
        }

    def write_values(_hass, metadata, rows):
        target = store.setdefault(metadata["statistic_id"], {})
        for row in rows:
            target[row["start"]] = dict(row)

    with (
        patch.object(client, "get_time_series_data", return_value=[series]),
        patch.object(
            client, "_get_hourly_stats_values_in_window", side_effect=read_values
        ),
        patch.object(
            client, "_get_recorded_price_stat_hours", side_effect=read_price_hours
        ),
        patch(
            "custom_components.fortum.api.client.async_add_external_statistics",
            side_effect=write_values,
        ),
    ):
        yield client, ids, store, series


async def test_repeated_import_and_late_pricing_preserve_sums(hourly_store):
    client, ids, store, series = hourly_store
    end = START + timedelta(hours=3)
    series.series = [_point(0, 1.0), _point(1, 0.0), _point(2, 2.0)]

    assert await client._record_hourly_data_stats("test", START, end) == 3
    assert store[ids.consumption][START + timedelta(hours=2)]["sum"] == 103.0
    assert len(store[ids.cost]) == 1  # Only the pre-existing baseline.
    assert ids.price not in store
    assert await client._record_hourly_data_stats("test", START, end) == 0

    # Pricing arrives later; missing energy must not replace recorded energy.
    series.series = [
        _point(0, None, priced=True, cost=0.0),
        _point(1, None, priced=True, cost=-0.5),
        _point(2, None, priced=True, cost=1.5),
    ]
    assert await client._record_hourly_data_stats("test", START, end) == 3
    assert store[ids.consumption][START + timedelta(hours=2)]["sum"] == 103.0
    assert store[ids.cost][START + timedelta(hours=2)]["sum"] == 51.0
    assert len(store[ids.price]) == 3
    assert await client._record_hourly_data_stats("test", START, end) == 0


async def test_omitted_unpriced_hour_is_preserved_before_recalculating(hourly_store):
    client, ids, store, series = hourly_store
    end = START + timedelta(hours=3)
    series.series = [_point(0, 1.0), _point(1, 2.0), _point(2, 3.0)]
    await client._record_hourly_data_stats("test", START, end)

    # The source temporarily omits one hour and corrects another.
    series.series = [_point(0, 1.0), _point(1, None), _point(2, 4.0)]
    assert await client._record_hourly_data_stats("test", START, end) == 1
    assert store[ids.consumption][START + timedelta(hours=1)]["state"] == 2.0
    assert store[ids.consumption][START + timedelta(hours=2)]["sum"] == 107.0
    assert await client._record_hourly_data_stats("test", START, end) == 0


async def test_partial_cost_omission_does_not_reduce_sum(hourly_store):
    client, ids, store, series = hourly_store
    end = START + timedelta(hours=3)
    series.series = [_point(i, 1.0, priced=True, cost=1.0) for i in range(3)]
    await client._record_hourly_data_stats("test", START, end)
    series.series[1].cost = None
    assert await client._record_hourly_data_stats("test", START, end) == 0
    assert store[ids.cost][START + timedelta(hours=2)]["sum"] == 53.0


async def test_price_and_cost_without_energy_do_not_create_consumption(hourly_store):
    client, ids, store, series = hourly_store
    series.series = [_point(energy=None, priced=True, cost=0.0)]
    assert (
        await client._record_hourly_data_stats(
            "test", START, START + timedelta(hours=1)
        )
        == 1
    )
    assert START not in store[ids.consumption]
    assert store[ids.price][START]["state"] == 0.0
    assert store[ids.cost][START]["sum"] == 50.0


async def test_multi_window_sync_revisits_delayed_prices(hourly_store):
    client, ids, store, series = hourly_store
    now = START + timedelta(days=35)
    hours = 35 * 24
    store[ids.price] = {START - timedelta(hours=1): {"state": 1.0}}
    series.series = [_point(i, 1.0) for i in range(hours)]
    points = (MeteringPoint(metering_point_no="test"),)

    with patch("custom_components.fortum.api.client.dt_util.utcnow", return_value=now):
        assert await client.sync_hourly_data_for_metering_points(points) == hours
        assert store[ids.consumption][now - timedelta(hours=1)]["sum"] == 100 + hours
        assert await client.sync_hourly_data_for_metering_points(points) == 0

        series.series = [_point(i, 1.0, priced=True, cost=0.5) for i in range(hours)]
        assert await client.sync_hourly_data_for_metering_points(points) == hours
        assert store[ids.cost][now - timedelta(hours=1)]["sum"] == 50 + hours * 0.5
        assert store[ids.consumption][now - timedelta(hours=1)]["sum"] == 100 + hours
        assert await client.sync_hourly_data_for_metering_points(points) == 0


async def test_backfill_without_cost_keeps_prior_cost_seed(hourly_store):
    client, ids, store, series = hourly_store
    series.series = [_point(0, 1.0)]
    now = START + timedelta(days=28)
    with (
        patch("custom_components.fortum.api.client.dt_util.utcnow", return_value=now),
        patch.object(
            client,
            "_find_first_recorded_price_gap_hour",
            side_effect=[START + timedelta(days=1), None],
        ),
        patch.object(
            client, "_recalculate_hourly_sums_until_end", return_value=0.0
        ) as recalculate,
    ):
        assert (
            await client.backfill_historical_price_gaps_for_metering_points(
                (MeteringPoint(metering_point_no="test"),)
            )
            == 1
        )

    assert recalculate.call_args.kwargs["consumption_seed_sum"] == 101.0
    assert recalculate.call_args.kwargs["cost_seed_sum"] == 50.0


@pytest.mark.parametrize("missing_metric", ["consumption", "cost"])
async def test_backfill_preserves_empty_stream_baseline_in_later_rows(
    hourly_store, missing_metric
):
    client, ids, store, series = hourly_store
    later_hour = START + timedelta(days=20)
    end = START + timedelta(days=28)
    for sid, baseline in ((ids.consumption, 100.0), (ids.cost, 50.0)):
        store[sid][later_hour] = {"state": 2.0, "sum": baseline + 2.0}
    series.series = [
        _point(
            0,
            None if missing_metric == "consumption" else 1.0,
            cost=None if missing_metric == "cost" else -0.5,
        )
    ]
    with (
        patch("custom_components.fortum.api.client.dt_util.utcnow", return_value=end),
        patch.object(
            client,
            "_find_first_recorded_price_gap_hour",
            side_effect=[START + timedelta(days=1), None],
        ),
    ):
        assert (
            await client.backfill_historical_price_gaps_for_metering_points(
                (MeteringPoint(metering_point_no="test"),)
            )
            == 1
        )

    assert store[ids.consumption][later_hour]["sum"] == (
        102.0 if missing_metric == "consumption" else 103.0
    )
    assert store[ids.cost][later_hour]["sum"] == (
        52.0 if missing_metric == "cost" else 51.5
    )
