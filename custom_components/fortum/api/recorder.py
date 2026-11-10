"""Acknowledged writes of the client's canonical hourly statistics to Recorder."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import timedelta
from math import isclose
from time import monotonic
from typing import TYPE_CHECKING

from homeassistant.components.recorder.db_schema import Statistics
from homeassistant.components.recorder.statistics import (
    UNIT_CLASS_TO_UNIT_CONVERTER,
    import_statistics,
    split_statistic_id,
    statistics_during_period,
    valid_statistic_id,
)
from homeassistant.components.recorder.tasks import CommitTask, RecorderTask
from homeassistant.helpers.recorder import get_instance

from ..exceptions import APIError

if TYPE_CHECKING:
    from homeassistant.components.recorder.core import Recorder
    from homeassistant.components.recorder.models import (
        StatisticData,
        StatisticMetaData,
    )
    from homeassistant.core import HomeAssistant


@dataclass(slots=True)
class _HourlyStatisticsTask(RecorderTask):
    """Retain the same acknowledgement when Recorder requests an import retry."""

    # Own the pre-import commit so a failure cannot bypass our acknowledgement.
    commit_before = False

    metadata: StatisticMetaData
    rows: list[StatisticData]
    done: asyncio.Future[None]
    deadline: float

    def run(self, instance: Recorder) -> None:
        """Import on the writer thread and acknowledge committed values only."""
        try:
            CommitTask().run(instance)
        except Exception as exc:
            self._finish(instance, exc)
            # Recorder must still recover its event session through its normal
            # error handler. Completing our future must not swallow that error.
            raise
        if monotonic() >= self.deadline:
            self._finish(instance, APIError("Hourly statistics write expired"))
            return
        try:
            if not import_statistics(instance, self.metadata, self.rows, Statistics):
                instance.queue_task(self)
                return
            # Recorder's import helper also returns True for a permanent database
            # error. Verify the result instead of mistaking that for success.
            self._verify(instance)
        except Exception as exc:
            self._finish(instance, exc)
        else:
            self._finish(instance, None)

    def _verify(self, instance: Recorder) -> None:
        """Read back committed states/sums through Recorder's statistics API."""
        statistic_id = self.metadata["statistic_id"]
        recorded = statistics_during_period(
            instance.hass,
            start_time=min(row["start"] for row in self.rows),
            end_time=max(row["start"] for row in self.rows) + timedelta(hours=1),
            statistic_ids={statistic_id},
            period="hour",
            units=None,
            types={"mean", "state", "sum"},
        )
        by_start = {row["start"]: row for row in recorded.get(statistic_id, [])}
        for expected in self.rows:
            actual = by_start.get(expected["start"].timestamp(), {})
            for field in ("state", "sum"):
                if field not in expected:
                    continue
                recorded_field = (
                    "mean"
                    if field == "state" and not self.metadata["has_sum"]
                    else field
                )
                value = actual.get(recorded_field)
                expected_value = expected.get(field)
                if (
                    not isinstance(value, int | float)
                    or not isinstance(expected_value, int | float)
                    or not isclose(value, expected_value, rel_tol=0.0, abs_tol=1e-9)
                ):
                    raise APIError("Recorder did not persist hourly statistics")

    def _finish(self, instance: Recorder, error: Exception | None) -> None:
        """Deliver completion on the event loop, including after caller timeout."""

        def complete() -> None:
            if self.done.done():
                return
            if error is None:
                self.done.set_result(None)
            else:
                self.done.set_exception(error)

        instance.hass.loop.call_soon_threadsafe(complete)


def queue_hourly_statistics(
    hass: HomeAssistant,
    metadata: StatisticMetaData,
    rows: list[StatisticData],
    *,
    timeout: float,
) -> asyncio.Future[None]:
    """Queue canonical hourly rows and return their durable-write acknowledgement.

    The client supplies normalized UTC hours, Fortum statistic IDs and metadata.
    All database writes and retry decisions remain owned by Recorder.
    """
    # Keep the validation previously provided by async_add_external_statistics.
    # This adapter accepts only the client's explicit metadata and hourly rows;
    # it does not implement HA's legacy metadata inference paths.
    statistic_id = metadata["statistic_id"]
    if not valid_statistic_id(statistic_id) or (
        split_statistic_id(statistic_id)[0] != metadata["source"]
    ):
        raise APIError("Invalid hourly statistic ID or source")
    if (unit_class := metadata["unit_class"]) is not None:
        converter = UNIT_CLASS_TO_UNIT_CONVERTER.get(unit_class)
        if (
            converter is None
            or metadata["unit_of_measurement"] not in converter.VALID_UNITS
        ):
            raise APIError("Invalid hourly statistics unit")
    for row in rows:
        start = row["start"]
        if (
            start.utcoffset() is None
            or start.minute
            or start.second
            or start.microsecond
        ):
            raise APIError("Hourly statistics require timezone-aware whole hours")
    done: asyncio.Future[None] = hass.loop.create_future()
    # Retrieve late exceptions even if the integration unloads after a timeout.
    done.add_done_callback(
        lambda future: None if future.cancelled() else future.exception()
    )
    get_instance(hass).queue_task(
        _HourlyStatisticsTask(metadata, rows, done, monotonic() + timeout)
    )
    return done
