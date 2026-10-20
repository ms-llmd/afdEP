# SPDX-License-Identifier: Apache-2.0
import math

import pytest

from benchmarks.screen import summarize


def test_summary_pairs_each_repeat_against_the_first_arm():
    processes = [
        {"median_us": {"masked": {1: 10.0, 8: 20.0}, "afd": {1: 9.0, 8: 22.0}}},
        {"median_us": {"masked": {1: 12.0, 8: 20.0}, "afd": {1: 12.0, 8: 18.0}}},
    ]
    table = summarize(processes, ["masked", "afd"])
    assert [row["tokens"] for row in table] == [1, 8]
    first = table[0]["afd"]
    assert first["median_us"] == 10.5
    assert first["paired_change_pct"] == pytest.approx([-10.0, 0.0])
    assert first["repeats_faster"] == 1
    assert first["geomean_change_pct"] == pytest.approx(100 * (math.sqrt(0.9) - 1))
    assert table[1]["masked"]["paired_change_pct"] == [0.0, 0.0]
