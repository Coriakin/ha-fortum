"""Hourly reconciliation against Home Assistant's threaded SQLite Recorder."""

import asyncio
from datetime import UTC, datetime, timedelta
from threading import Event as ThreadEvent
from unittest.mock import patch

import pytest
from homeassistant.components.recorder.core import Recorder
from homeassistant.components.recorder.tasks import RecorderTask
from homeassistant.core import HomeAssistant
from homeassistant.helpers.recorder import DATA_INSTANCE, DATA_RECORDER, RecorderData

from custom_components.fortum.api.client import FortumAPIClient
from custom_components.fortum.api.recorder import _HourlyStatisticsTask
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
    original_run = _HourlyStatisticsTask.run
    original_enqueue = client._queue_hourly_statistics

    def hold_import(task, instance):
        writing.set()
        assert release.wait(10), "Test did not release in-progress import"
        original_run(task, instance)

    class HoldWriter(RecorderTask):
        def run(self, instance):
            instance.hass.loop.call_soon_threadsafe(entered.set)
            assert release.wait(10), "Test did not release Recorder writer"

    def enqueue(metadata, rows):
        original_enqueue(metadata, rows)
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
                    _HourlyStatisticsTask,
                    "run",
                    hold_import if write_in_progress else original_run,
                ),
                patch.object(
                    client, "get_time_series_data", return_value=[_series([_point(0)])]
                ),
                patch.object(
                    client,
                    "_queue_hourly_statistics",
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


@pytest.mark.parametrize("retry_tail", [False, True])
async def test_backfill_without_cost_retains_persisted_baseline(
    recorder_client, retry_tail
):
    """An empty cost window must not reset sums in the later persisted tail."""
    from homeassistant.components.recorder import statistics
    from sqlalchemy.exc import OperationalError

    client, _recorder = recorder_client
    ids = client._build_hourly_statistic_ids("test")
    original_import = statistics._import_statistics_with_session
    failures = 0

    def retry_cost_tail(instance, session, metadata, rows, table):
        nonlocal failures
        if retry_tail and failures == 0 and metadata["statistic_id"] == ids.cost:
            failures += 1
            raise OperationalError("test import", {}, Exception(1213, "test deadlock"))
        return original_import(instance, session, metadata, rows, table)

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
            patch.object(
                statistics,
                "_import_statistics_with_session",
                side_effect=retry_cost_tail,
            ),
            patch(
                "homeassistant.components.recorder.util._is_retryable_error",
                return_value=True,
            ),
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
        assert failures == int(retry_tail)


@pytest.mark.parametrize("failures", [1, 3])
async def test_database_retries_finish_before_the_next_window(
    recorder_client, failures
):
    """Real Recorder retries retain their acknowledgement behind other tasks."""
    from homeassistant.components.recorder import statistics
    from sqlalchemy.exc import OperationalError

    client, recorder = recorder_client
    entered = asyncio.Event()
    release = ThreadEvent()
    original_import = statistics._import_statistics_with_session
    attempts = 0

    def retry_import(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts <= failures:
            raise OperationalError("test import", {}, Exception(1213, "test deadlock"))
        if attempts == failures + 1:
            recorder.hass.loop.call_soon_threadsafe(entered.set)
            assert release.wait(5)
        return original_import(*args, **kwargs)

    end = START + timedelta(days=14, hours=1)
    series = _series([_point(0), _point(14 * 24)])
    sync = None
    try:
        async with asyncio.timeout(10):
            with (
                patch.object(
                    client, "get_time_series_data", return_value=[series]
                ) as fetch,
                patch.object(
                    statistics,
                    "_import_statistics_with_session",
                    side_effect=retry_import,
                ),
                # Exercise HA's retry decorator and our task on SQLite; the
                # production retry classification is MySQL-specific.
                patch(
                    "homeassistant.components.recorder.util._is_retryable_error",
                    return_value=True,
                ),
            ):
                sync = asyncio.create_task(client._sync_hourly_data("test", START, end))
                await entered.wait()
                assert not sync.done()
                assert fetch.await_count == 1
                assert client._last_hourly_stats_digest is None
                release.set()
                assert await sync == 2
                assert fetch.await_count == 2
            ids = client._build_hourly_statistic_ids("test")
            assert (
                await client._get_hourly_stat_sum_before_hour(ids.consumption, end)
                == 2.0
            )
    finally:
        release.set()
        if sync is not None:
            await sync


@pytest.mark.parametrize("in_progress", [False, True])
async def test_timeout_retains_pending_write_and_blocks_new_reads(
    recorder_client, in_progress
):
    """A timed-out writer cannot race a later import or suppress it via digest."""
    from custom_components.fortum.exceptions import APIError

    client, recorder = recorder_client
    entered = asyncio.Event()
    release = ThreadEvent()
    original_verify = _HourlyStatisticsTask._verify

    class HoldWriter(RecorderTask):
        def run(self, instance):
            instance.hass.loop.call_soon_threadsafe(entered.set)
            assert release.wait(5)

    def hold_verify(task, instance):
        # The import has committed but its acknowledgement is still in flight.
        instance.hass.loop.call_soon_threadsafe(entered.set)
        assert release.wait(5)
        original_verify(task, instance)

    if not in_progress:
        recorder.queue_task(HoldWriter())
    series = _series([_point(0)])
    end = START + timedelta(hours=1)
    try:
        with (
            patch.object(
                client, "get_time_series_data", return_value=[series]
            ) as fetch,
            patch.object(
                _HourlyStatisticsTask,
                "_verify",
                hold_verify if in_progress else original_verify,
            ),
            patch(
                "custom_components.fortum.api.client.HOURLY_STATISTICS_WRITE_TIMEOUT_SECONDS",
                0.05,
            ),
        ):
            if not in_progress:
                await entered.wait()
            with pytest.raises(APIError, match="Timed out"):
                await client._record_hourly_data_stats("test", START, end)
            assert entered.is_set()
            assert client._last_hourly_stats_digest is None
            assert client._pending_hourly_writes
            assert all(not done.cancelled() for done in client._pending_hourly_writes)
            with pytest.raises(APIError, match="Timed out"):
                await client._record_hourly_data_stats("test", START, end)
            assert fetch.await_count == 1
            release.set()
            results = await asyncio.gather(
                *client._pending_hourly_writes, return_exceptions=True
            )
            if not in_progress:
                assert isinstance(results[0], APIError)  # Expired before writing.
                with pytest.raises(APIError, match="Failed to persist"):
                    await client._record_hourly_data_stats("test", START, end)
                assert fetch.await_count == 1
            else:
                assert results == [None]

        series.series = [_point(0, 3.0)]
        with patch.object(client, "get_time_series_data", return_value=[series]):
            assert await client._record_hourly_data_stats("test", START, end) == 1
            assert await client._record_hourly_data_stats("test", START, end) == 0
        ids = client._build_hourly_statistic_ids("test")
        assert (
            await client._get_hourly_stat_sum_before_hour(ids.consumption, end) == 3.0
        )
    finally:
        release.set()
        await asyncio.gather(*client._pending_hourly_writes, return_exceptions=True)


async def test_permanent_database_failure_is_not_acknowledged(recorder_client):
    """HA returns True after a permanent error; read-back must detect lost rows."""
    from homeassistant.components.recorder import statistics
    from sqlalchemy.exc import OperationalError

    from custom_components.fortum.exceptions import APIError

    client, _recorder = recorder_client
    series = _series([_point(0)])
    end = START + timedelta(hours=1)
    with patch.object(client, "get_time_series_data", return_value=[series]):
        with patch.object(
            statistics,
            "_import_statistics_with_session",
            side_effect=OperationalError(
                "test import", {}, Exception("permanent test error")
            ),
        ):
            with pytest.raises(APIError, match="Failed to persist"):
                await client._record_hourly_data_stats("test", START, end)
        assert client._last_hourly_stats_digest is None
        assert client._pending_hourly_writes == []
        assert await client._record_hourly_data_stats("test", START, end) == 1
        assert await client._record_hourly_data_stats("test", START, end) == 0


@pytest.mark.parametrize("price", [0.0, -0.5])
@pytest.mark.parametrize("with_temperature", [False, True])
async def test_price_only_import_is_idempotent_without_cumulative_metadata(
    recorder_client, price, with_temperature
):
    """Reproduce HA omitting state when only mean statistics exist."""
    from homeassistant.components.recorder.statistics import get_metadata

    from custom_components.fortum.models import TemperatureReading

    client, recorder = recorder_client
    ids = client._build_hourly_statistic_ids("test")
    point = _point(0, energy=None)
    point.price = Price(price, price, 0.0, 25.0)
    point.temperature_reading = TemperatureReading(5.0) if with_temperature else None
    series = _series([point])
    end = START + timedelta(hours=1)
    with patch.object(client, "get_time_series_data", return_value=[series]):
        assert await client._record_hourly_data_stats("test", START, end) == 1
        metadata = await recorder.async_add_executor_job(
            lambda: get_metadata(recorder.hass, statistic_ids=set(ids.as_tuple()))
        )
        assert set(metadata) == (
            {ids.price, ids.temperature} if with_temperature else {ids.price}
        )
        assert await client._record_hourly_data_stats("test", START, end) == 0

        # Correct two streams in the same hour, but count the hour only once.
        point.price = Price(price + 1.0, price + 1.0, 0.0, 25.0)
        if with_temperature:
            point.temperature_reading = TemperatureReading(6.0)
        assert await client._record_hourly_data_stats("test", START, end) == 1
        assert await client._record_hourly_data_stats("test", START, end) == 0

        # A later empty response preserves mean streams without adding readings.
        series.series = []
        assert await client._record_hourly_data_stats("test", START, end) == 0
        assert await client._get_hourly_stats_values_in_window(
            {ids.price}, START, end, value_type="mean"
        ) == {ids.price: {START: price + 1.0}}

        # Consumption may arrive later, with pricing omitted again.
        series.series = [_point(0, energy=3.0)]
        assert await client._record_hourly_data_stats("test", START, end) == 1
        assert await client._record_hourly_data_stats("test", START, end) == 0
        assert (
            await client._get_hourly_stat_sum_before_hour(ids.consumption, end) == 3.0
        )
        assert await client._get_hourly_stats_values_in_window(
            {ids.price}, START, end, value_type="mean"
        ) == {ids.price: {START: price + 1.0}}
        if with_temperature:
            assert await client._get_hourly_stats_values_in_window(
                {ids.temperature}, START, end, value_type="mean"
            ) == {ids.temperature: {START: 6.0}}


async def test_event_commit_failure_acknowledges_write_and_allows_recovery(
    recorder_client,
):
    """A failure before the statistics import must not orphan its acknowledgement."""
    from sqlalchemy.exc import SQLAlchemyError

    from custom_components.fortum.exceptions import APIError

    client, recorder = recorder_client
    original_process = recorder._process_one_task_or_event_or_recover
    original_commit = recorder._commit_event_session_or_retry
    processing_hourly_task = False
    processed = asyncio.Event()
    failures = 0

    def process(task):
        nonlocal processing_hourly_task
        processing_hourly_task = isinstance(task, _HourlyStatisticsTask)
        try:
            original_process(task)
        finally:
            if processing_hourly_task:
                recorder.hass.loop.call_soon_threadsafe(processed.set)
            processing_hourly_task = False

    def fail_commit_once():
        nonlocal failures
        if processing_hourly_task and failures == 0:
            failures += 1
            raise SQLAlchemyError("test event-session commit failure")
        original_commit()

    series = _series([_point(0)])
    end = START + timedelta(hours=1)
    try:
        with (
            patch.object(client, "get_time_series_data", return_value=[series]),
            patch.object(
                recorder, "_process_one_task_or_event_or_recover", side_effect=process
            ),
            patch.object(
                recorder, "_commit_event_session_or_retry", side_effect=fail_commit_once
            ),
            patch.object(
                recorder, "_reopen_event_session", wraps=recorder._reopen_event_session
            ) as recover,
            patch(
                "custom_components.fortum.api.client.HOURLY_STATISTICS_WRITE_TIMEOUT_SECONDS",
                0.1,
            ),
        ):
            with pytest.raises(APIError, match="Failed to persist"):
                await client._record_hourly_data_stats("test", START, end)
            assert failures == 1
            assert client._pending_hourly_writes == []
            assert client._last_hourly_stats_digest is None

            # Recorder must still perform its normal event-session recovery.
            async with asyncio.timeout(5):
                await processed.wait()
            recover.assert_called_once()
            series.series = [_point(0, energy=2.0)]
            assert await client._record_hourly_data_stats("test", START, end) == 1
            assert await client._record_hourly_data_stats("test", START, end) == 0
    finally:
        # Clean up the orphan on the unfixed implementation, so the regression
        # fails promptly rather than leaving an unfinished test future behind.
        for done in client._pending_hourly_writes:
            done.cancel()
