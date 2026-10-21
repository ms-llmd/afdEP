# SPDX-License-Identifier: Apache-2.0
"""Alternate single-arm benchmark processes and pair them by repeat.

Each repeat launches one fresh four-rank process per arm, rotating the order.
A repeat's process medians form one paired comparison against the first arm.

Example: python -m benchmarks.screen --repeats 4 --output /tmp/screen.json -- \
    --payload bf16 --routing random --tokens 1 32 128 2048
"""

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path

# Arms are Config overrides. Masked keeps DeepEP's union protocol and dummy slots.
DEFAULT_ARMS = {"masked": {"role_aware": False}, "afd": {}}
ENV_KEY = "_env"
# "_tree" runs the arm from another source tree, imported alone (PYTHONSAFEPATH);
# "_digest" is the native source digest every rank of the arm must report.
TREE_KEY, DIGEST_KEY = "_tree", "_digest"
LAUNCH = ["torchrun", "--standalone", "--nproc-per-node"]
PROCESS_TIMEOUT_S = 900


def run_arm(name, overrides, repeat, args, passthrough):
    """Run one fresh process; an arm's optional "_env" applies to that process."""
    overrides = dict(overrides)
    env = os.environ | overrides.pop(ENV_KEY, {})
    tree, digest = overrides.pop(TREE_KEY, None), overrides.pop(DIGEST_KEY, None)
    cwd = None
    if tree is not None:
        cwd = str(Path(tree).resolve())
        env |= {"PYTHONPATH": cwd, "PYTHONSAFEPATH": "1"}
    output = args.output.parent / f"{args.output.stem}-{name}-r{repeat}.json"
    command = [
        *LAUNCH,
        str(args.nproc),
        "-m",
        "benchmarks.comm",
        *passthrough,
        "--arms",
        json.dumps({name: overrides}),
        "--output",
        str(output),
    ]
    log = output.with_suffix(".log")
    with log.open("w") as stream:
        code = subprocess.run(
            command,
            stdout=stream,
            stderr=subprocess.STDOUT,
            timeout=PROCESS_TIMEOUT_S,
            env=env,
            cwd=cwd,
        ).returncode
    if code:
        raise SystemExit(f"{name} repeat {repeat} failed ({code}); see {log}")
    result = json.loads(output.read_text())
    digests = {rank["native_source_sha256"] for rank in result["metadata"]}
    if len(digests) != 1 or (digest is not None and digests != {digest}):
        raise SystemExit(f"{name} repeat {repeat}: native digests {sorted(digests)}")
    cells = result["cells"]
    medians = {cell["tokens"]: cell["summary"][name]["median_us"] for cell in cells}
    return medians, digests.pop()


def summarize(processes, names):
    """Per token extent: median process value and paired change vs names[0]."""
    control = names[0]
    table = []
    for n in sorted(processes[0]["median_us"][control]):
        row = {"tokens": n}
        for name in names:
            values = [p["median_us"][name][n] for p in processes]
            changes = [
                100 * (p["median_us"][name][n] / p["median_us"][control][n] - 1)
                for p in processes
            ]
            row[name] = {
                "median_us": statistics.median(values),
                "paired_change_pct": changes,
                "geomean_change_pct": 100
                * (
                    math.exp(statistics.fmean(math.log1p(c / 100) for c in changes)) - 1
                ),
                "repeats_faster": sum(c < 0 for c in changes),
            }
        table.append(row)
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", type=json.loads, default=DEFAULT_ARMS)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--nproc", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args, passthrough = parser.parse_known_args()
    passthrough = [value for value in passthrough if value != "--"]
    if args.repeats < 1 or not isinstance(args.arms, dict) or not args.arms:
        parser.error("repeats must be positive and --arms a nonempty JSON object")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    names = list(args.arms)
    processes = []
    for repeat in range(args.repeats):
        order = names[repeat % len(names) :] + names[: repeat % len(names)]
        runs = {
            name: run_arm(name, args.arms[name], repeat, args, passthrough)
            for name in order
        }
        results = {name: medians for name, (medians, _) in runs.items()}
        digests = {name: digest for name, (_, digest) in runs.items()}
        processes.append(
            {"order": order, "median_us": results, "native_source_sha256": digests}
        )
        print(json.dumps({"repeat": repeat} | results), flush=True)
    table = summarize(processes, names)
    args.output.write_text(
        json.dumps(
            {
                "arms": args.arms,
                "passthrough": passthrough,
                "processes": processes,
                "table": table,
                "pod": os.environ.get("HOSTNAME"),
                "python": sys.version,
            },
            indent=1,
        )
        + "\n"
    )
    for row in table:
        print(
            json.dumps(
                {"tokens": row["tokens"]}
                | {
                    name: [
                        round(row[name]["median_us"], 2),
                        round(row[name]["geomean_change_pct"], 2),
                        row[name]["repeats_faster"],
                    ]
                    for name in names
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
