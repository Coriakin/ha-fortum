"""Hourly reconciliation against Home Assistant's threaded SQLite Recorder."""

import asyncio
from datetime import UTC, datetime, timedelta
from threading import Event as ThreadEvent
from unittest.mock import patch

import pytest
from homeassistant.components.recorder.core import Recorder
from homeassistant.components.recorder.statistics import async_add_external_statistics
from homeassistant.components.recorder.tasks import ImportStatisticsTask, RecorderTask
from homeassistant.core import HomeAssistant
from homeassistant.helpers.recorder import DATA_INSTANCE, DATA_RECORDER, RecorderData

from custom_components.fortum.api.client import FortumAPIClient
from custom_components.fortum.models import (
    CostDataPoint,
    EnergyDataPoint,
    MeteringPoint,
    Price,
    TimeSeries,
    TimeSeriesDataPoint,
)

START = datetime(2026, 1, 1, tzinfo=UTC)


@pytest.fixture
async def recorder_client(tmp_path, mock_auth_client):
    """Use a temporary database, real writer thread, and real query executor."""
    hass = HomeAssistant(str(tmp_path))
    hass.data[DATA_RECORDER] = RecorderData()
    recorder = Recorder(
        hass=hass,
        auto_purge=False,
        auto_repack=False,
        keep_days=10,
        commit_interval=0,
        uri=f"sqlite:///{tmp_path / 'recorder.db'}",
        db_max_retries=1,
        db_retry_wait=0,
        entity_filter=None,
        exclude_event_types=set(),
    )
    hass.data[DATA_INSTANCE] = recorder
    recorder.async_initialize()
    recorder.async_register()
    recorder.start()
    try:
        async with asyncio.timeout(15):
            assert await recorder.async_db_ready
            await hass.async_start()
            await recorder.async_block_till_done()
        yield FortumAPIClient(hass, mock_auth_client), recorder
    finally:
        await hass.async_stop()


def _series(points):
    return TimeSeries(
        delivery_site_category="CONSUMPTION",
        measurement_unit="kWh",
        metering_point_no="test",
        price_unit="öre/kWh",
        cost_unit="SEK",
        temperature_unit="celsius",
        series=points,
    )


def _point(hour, energy=1.0, cost=None):
    return TimeSeriesDataPoint(
        at_utc=START + timedelta(hours=hour),
        energy=[] if energy is None else [EnergyDataPoint(energy, "ENERGY")],
        cost=None if cost is None else [CostDataPoint(cost, cost, "COST")],
        price=None if cost is None else Price(0.0, 0.0, 0.0, 25.0),
        temperature_reading=None,
    )


@pytest.mark.parametrize("write_in_progress", [False, True])
async def test_import_waits_for_queued_recorder_writes(
    recorder_client, write_in_progress
):
    """A completed import must be visible to the next window's baseline query."""
    client, recorder = recorder_client
    entered = asyncio.Event()
    queued = asyncio.Event()
    release = ThreadEvent()
    writing = ThreadEvent()
    original_run = ImportStatisticsTask.run

    def hold_import(task, instance):
        writing.set()
        assert release.wait(10), "Test did not release in-progress import"
        original_run(task, instance)

    class HoldWriter(RecorderTask):
        def run(self, instance):
            instance.hass.loop.call_soon_threadsafe(entered.set)
            assert release.wait(10), "Test did not release Recorder writer"

    def enqueue(hass, metadata, rows):
        async_add_external_statistics(hass, metadata, rows)
        if write_in_progress:
            # Force the writer to dequeue the task before the importer waits.
            assert writing.wait(10), "Recorder did not start the import"
        queued.set()

    if write_in_progress:
        entered.set()
    else:
        recorder.queue_task(HoldWriter())
    import_task = None
    try:
        async with asyncio.timeout(10):
            await entered.wait()
            with (
                patch.object(
                    ImportStatisticsTask,
                    "run",
                    hold_import if write_in_progress else original_run,
                ),
                patch.object(
                    client, "get_time_series_data", return_value=[_series([_point(0)])]
                ),
                patch(
                    "custom_components.fortum.api.client.async_add_external_statistics",
                    side_effect=enqueue,
                ),
            ):
                import_task = asyncio.create_task(
                    client._record_hourly_data_stats(
                        "test", START, START + timedelta(hours=1)
                    )
                )
                await queued.wait()
                assert not import_task.done()
                release.set()
                assert await import_task == 1
            ids = client._build_hourly_statistic_ids("test")
            assert (
                await client._get_hourly_stat_sum_before_hour(
                    ids.consumption, START + timedelta(hours=1)
                )
                == 1.0
            )
    finally:
        release.set()
        if import_task is not None:
            await import_task
        await recorder.async_block_till_done()


async def test_multi_window_reconciliation_with_real_recorder(recorder_client):
    """Exercise persisted baselines, corrections, omissions and delayed pricing."""
    client, recorder = recorder_client
    ids = client._build_hourly_statistic_ids("test")
    hours = 35 * 24
    end = START + timedelta(hours=hours)
    series = _series([_point(-1, 4.0, 2.0)])
    with patch.object(client, "get_time_series_data", return_value=[series]):
        await client._record_hourly_data_stats(
            "test", START - timedelta(hours=1), START
        )
        await recorder.async_block_till_done()

        series.series = [_point(i) for i in range(hours)]
        points = (MeteringPoint(metering_point_no="test"),)
        with patch(
            "custom_components.fortum.api.client.dt_util.utcnow", return_value=end
        ):
            assert await client.sync_hourly_data_for_metering_points(points) == hours
            assert await client.sync_hourly_data_for_metering_points(points) == 0

            # Correct an unpriced historical hour and omit its neighbor entirely.
            series.series[0].energy = [EnergyDataPoint(3.0, "ENERGY")]
            del series.series[1]
            assert await client.sync_hourly_data_for_metering_points(points) == 1

            # Pricing arrives without energy; retain every consumption contribution.
            series.series = [
                _point(i, None, -0.5 if i == 0 else 0.5) for i in range(hours)
            ]
            assert await client.sync_hourly_data_for_metering_points(points) == hours
            assert await client.sync_hourly_data_for_metering_points(points) == 0

        sums = await client._get_hourly_stats_values_in_window(
            {ids.consumption, ids.cost}, START, end, value_type="sum"
        )
        assert len(sums[ids.consumption]) == hours
        assert len(sums[ids.cost]) == hours
        assert sums[ids.consumption][end - timedelta(hours=1)] == 4.0 + hours + 2.0
        assert sums[ids.cost][end - timedelta(hours=1)] == 2.0 + hours * 0.5 - 1.0


async def test_backfill_without_cost_retains_persisted_baseline(recorder_client):
    """An empty cost window must not reset sums in the later persisted tail."""
    client, _recorder = recorder_client
    ids = client._build_hourly_statistic_ids("test")
    later_hour = START + timedelta(days=20)
    end = START + timedelta(days=28)
    series = _series([_point(-1, 4.0, 2.0)])
    with patch.object(client, "get_time_series_data", return_value=[series]):
        await client._record_hourly_data_stats(
            "test", START - timedelta(hours=1), START
        )
        series.series = [_point(20 * 24, 2.0, 1.0)]
        await client._record_hourly_data_stats("test", later_hour, end)

        series.series = [_point(0, 1.0)]
        with (
            patch(
                "custom_components.fortum.api.client.dt_util.utcnow", return_value=end
            ),
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

        sums = await client._get_hourly_stats_values_in_window(
            {ids.consumption, ids.cost}, START, end, value_type="sum"
        )
        assert sums[ids.consumption] == {START: 5.0, later_hour: 7.0}
        assert sums[ids.cost] == {later_hour: 3.0}
