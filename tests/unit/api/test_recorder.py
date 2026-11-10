"""Failure contracts for acknowledged hourly writes."""

import asyncio
from datetime import UTC, datetime
from threading import get_ident
from unittest.mock import Mock, patch

import pytest
from homeassistant.helpers import frame
from homeassistant.helpers.update_coordinator import UpdateFailed

from custom_components.fortum.api.client import FortumAPIClient
from custom_components.fortum.api.recorder import queue_hourly_statistics
from custom_components.fortum.coordinators.hourly_consumption import (
    HourlyConsumptionSyncCoordinator,
)
from custom_components.fortum.exceptions import APIError


@pytest.mark.parametrize("invalid", ["source", "id", "unit", "naive", "minute"])
async def test_invalid_hourly_write_is_rejected_before_queuing(
    mock_hass, mock_auth_client, invalid
):
    client = FortumAPIClient(mock_hass, mock_auth_client)
    metadata = client._build_hourly_statistic_metadata(
        statistic_id="fortum:hourly_consumption_test",
        name="Test consumption",
        unit_of_measurement="kWh",
        unit_class="energy",
        has_sum=True,
    )
    row = {"start": datetime(2026, 1, 1, tzinfo=UTC), "state": 1.0, "sum": 1.0}
    if invalid == "source":
        metadata["source"] = "other"
    elif invalid == "id":
        metadata["statistic_id"] = "invalid"
    elif invalid == "unit":
        metadata["unit_of_measurement"] = "invalid"
    elif invalid == "naive":
        row["start"] = datetime(2026, 1, 1)
    else:
        row["start"] = datetime(2026, 1, 1, 0, 1, tzinfo=UTC)
    with patch("custom_components.fortum.api.recorder.get_instance") as get_recorder:
        with pytest.raises(APIError):
            queue_hourly_statistics(mock_hass, metadata, [row], timeout=60)
        get_recorder.assert_not_called()


async def test_timeout_releases_coordinator_lock(mock_hass, mock_auth_client):
    """A stopped Recorder must not retain the integration's sync lock."""
    client = FortumAPIClient(mock_hass, mock_auth_client)
    done = asyncio.get_running_loop().create_future()
    client._pending_hourly_writes = [done]
    mock_hass.loop_thread_id = get_ident()
    frame.async_setup(mock_hass)
    coordinator = HourlyConsumptionSyncCoordinator(mock_hass, client, Mock())

    async def stalled_sync(_points):
        await client._wait_for_hourly_statistics_writes()
        return 0

    with (
        patch.object(
            client, "sync_hourly_data_for_metering_points", side_effect=stalled_sync
        ),
        patch(
            "custom_components.fortum.api.client.HOURLY_STATISTICS_WRITE_TIMEOUT_SECONDS",
            0.01,
        ),
    ):
        with pytest.raises(UpdateFailed, match="Timed out"):
            await coordinator._async_update_data()
        assert not coordinator._statistics_operation_lock.locked()
        assert coordinator.last_statistics_sync is None
        assert not done.cancelled()
        done.set_result(None)
        await client._wait_for_hourly_statistics_writes()
        assert client._pending_hourly_writes == []


async def test_cancellation_keeps_inflight_writes_owned(mock_hass, mock_auth_client):
    client = FortumAPIClient(mock_hass, mock_auth_client)
    done = asyncio.get_running_loop().create_future()
    client._pending_hourly_writes = [done]
    entered = asyncio.Event()

    async def wait():
        entered.set()
        await client._wait_for_hourly_statistics_writes()

    waiter = asyncio.create_task(wait())
    await entered.wait()
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert client._pending_hourly_writes == [done]
    assert not done.cancelled()
    done.set_result(None)
    await client._wait_for_hourly_statistics_writes()
    assert client._pending_hourly_writes == []


async def test_failed_write_does_not_release_other_pending_writes(
    mock_hass, mock_auth_client
):
    client = FortumAPIClient(mock_hass, mock_auth_client)
    failed = asyncio.get_running_loop().create_future()
    pending = asyncio.get_running_loop().create_future()
    failed.set_exception(APIError("test failure"))
    client._pending_hourly_writes = [failed, pending]
    with patch(
        "custom_components.fortum.api.client.HOURLY_STATISTICS_WRITE_TIMEOUT_SECONDS",
        0.01,
    ):
        with pytest.raises(APIError, match="Timed out"):
            await client._wait_for_hourly_statistics_writes()
    assert client._pending_hourly_writes == [failed, pending]
    assert not pending.cancelled()
    pending.set_result(None)
    with pytest.raises(APIError, match="Failed to persist"):
        await client._wait_for_hourly_statistics_writes()
    assert client._pending_hourly_writes == []


async def test_expired_database_retry_is_not_written_again(mock_hass, mock_auth_client):
    from custom_components.fortum.api.recorder import _HourlyStatisticsTask

    client = FortumAPIClient(mock_hass, mock_auth_client)
    metadata = client._build_hourly_statistic_metadata(
        statistic_id="fortum:hourly_consumption_test",
        name="Test consumption",
        unit_of_measurement="kWh",
        unit_class="energy",
        has_sum=True,
    )
    done = asyncio.get_running_loop().create_future()
    task = _HourlyStatisticsTask(
        metadata,
        [{"start": datetime(2026, 1, 1, tzinfo=UTC), "state": 1.0, "sum": 1.0}],
        done,
        deadline=1.0,
    )
    recorder = Mock()
    recorder.hass.loop = asyncio.get_running_loop()
    with (
        patch(
            "custom_components.fortum.api.recorder.import_statistics",
            return_value=False,
        ) as write,
        patch(
            "custom_components.fortum.api.recorder.monotonic", side_effect=[0.0, 2.0]
        ),
    ):
        task.run(recorder)
        recorder.queue_task.assert_called_once_with(task)
        assert not done.done()
        task.run(recorder)
        with pytest.raises(APIError, match="expired"):
            await done
        write.assert_called_once()


async def test_event_commit_cannot_extend_hourly_write_deadline(
    mock_hass, mock_auth_client
):
    """An event-session commit can consume the remaining statistics deadline."""
    from custom_components.fortum.api.recorder import _HourlyStatisticsTask

    client = FortumAPIClient(mock_hass, mock_auth_client)
    metadata = client._build_hourly_statistic_metadata(
        statistic_id="fortum:hourly_consumption_test",
        name="Test consumption",
        unit_of_measurement="kWh",
        unit_class="energy",
        has_sum=True,
    )
    done = asyncio.get_running_loop().create_future()
    task = _HourlyStatisticsTask(
        metadata,
        [{"start": datetime(2026, 1, 1, tzinfo=UTC), "state": 1.0, "sum": 1.0}],
        done,
        deadline=1.0,
    )
    recorder = Mock()
    recorder.hass.loop = asyncio.get_running_loop()
    elapsed = 0.0

    def commit():
        nonlocal elapsed
        elapsed = 2.0

    recorder._commit_event_session_or_retry.side_effect = commit
    with (
        patch("custom_components.fortum.api.recorder.import_statistics") as write,
        patch(
            "custom_components.fortum.api.recorder.monotonic",
            side_effect=lambda: elapsed,
        ),
    ):
        task.run(recorder)
        with pytest.raises(APIError, match="expired"):
            await done
        recorder._commit_event_session_or_retry.assert_called_once()
        write.assert_not_called()
        recorder.queue_task.assert_not_called()
