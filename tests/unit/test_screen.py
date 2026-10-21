# SPDX-License-Identifier: Apache-2.0
import json
import math
import subprocess
from argparse import Namespace

import pytest

from benchmarks import screen
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


def fake_run(digests, calls):
    def run(command, *, stdout, stderr, timeout, env, cwd):
        output = command[command.index("--output") + 1]
        name = next(iter(json.loads(command[command.index("--arms") + 1])))
        calls.append({"env": env, "cwd": cwd})
        cells = [{"tokens": 1, "summary": {name: {"median_us": 5.0}}}]
        metadata = [{"native_source_sha256": digest} for digest in digests]
        with open(output, "w") as stream:
            json.dump({"cells": cells, "metadata": metadata}, stream)
        return subprocess.CompletedProcess(command, 0)

    return run


def test_tree_arm_imports_its_own_tree_and_checks_its_digest(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(screen.subprocess, "run", fake_run(["d", "d"], calls))
    args = Namespace(output=tmp_path / "s.json", nproc=4)
    arm = {"_tree": str(tmp_path), "_digest": "d", "_env": {"A": "1"}}
    medians, digest = screen.run_arm("head", arm, 0, args, [])
    assert medians == {1: 5.0} and digest == "d"
    assert calls[0]["cwd"] == str(tmp_path.resolve())
    env = calls[0]["env"]
    assert env["PYTHONPATH"] == str(tmp_path.resolve()) and env["PYTHONSAFEPATH"] == "1"
    assert env["A"] == "1"
    monkeypatch.setattr(screen.subprocess, "run", fake_run(["d", "e"], calls))
    with pytest.raises(SystemExit, match="native digests"):
        screen.run_arm("head", {"_digest": "d"}, 0, args, [])
