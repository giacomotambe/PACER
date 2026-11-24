"""Analisi di una o piu' prove hardware (rev. 2026-11-04), stesse metriche del
benchmark simulato 13_closed_loop_benchmark.

Input per prova (scritti dai nodi ROS2):
  --tracks  CSV di pacer_qualisys_bridge (tracks_log_csv): tick, stamp, id, x, y, fresh
  --scores  CSV di pacer_calibration (score_log_csv): t_issue, k, ped_id, score, q_used, method, stamp
  --status  CSV di pacer_mpc_solver (status_log_csv): stamp, status, v, omega, min_margin, ...

Uso:
  python analyze_hw_run.py --tracks run03_tracks.csv --scores run03_scores.csv \
      --status run03_status.csv --robot-id omnicar --L 0.6
"""
import argparse
import csv
import math
from bisect import bisect_right
from collections import Counter, defaultdict

import numpy as np

VERIFIED = {"optimal", "fallback_stop", "fallback_evasive"}


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, c - h), min(1.0, c + h)


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def analyze(tracks, scores, status, robot_id, L):
    out = {}
    # posizioni per tick
    by_tick = defaultdict(dict)
    stamp_of = {}
    for r in tracks:
        if int(r["fresh"]):
            by_tick[int(r["tick"])][r["id"]] = (float(r["x"]), float(r["y"]))
            stamp_of[int(r["tick"])] = float(r["stamp"])
    st = sorted((float(r["stamp"]), r["status"]) for r in status) if status else []
    st_t = [s for s, _ in st]

    dmins, events = [], Counter()
    for t in sorted(by_tick):
        pos = by_tick[t]
        if robot_id not in pos:
            continue
        rx, ry = pos[robot_id]
        d = [math.hypot(x - rx, y - ry) for pid, (x, y) in pos.items() if pid != robot_id]
        if not d:
            continue
        dmin = min(d)
        dmins.append(dmin)
        if dmin < L:
            # stato dell'ultimo ciclo MPC concluso prima di questo istante
            i = bisect_right(st_t, stamp_of[t]) - 1
            prev = st[i][1] if i >= 0 else "unknown"
            events["after_verified" if prev in VERIFIED else f"after_{prev}"] += 1
    out["ticks_with_robot"] = len(dmins)
    out["min_dist_m"] = float(np.min(dmins)) if dmins else float("nan")
    out["p05_dist_m"] = float(np.percentile(dmins, 5)) if dmins else float("nan")
    out["events_below_L"] = dict(events)

    if status:
        c = Counter(r["status"] for r in status)
        n = sum(c.values())
        out["status_pct"] = {k: round(100 * v / n, 2) for k, v in c.items()}
        ms = np.array([float(r["solve_ms"]) for r in status])
        out["solve_ms_median"] = float(np.median(ms))
        out["solve_ms_p95"] = float(np.percentile(ms, 95))

    if scores:
        by_m = defaultdict(lambda: [0, 0])
        for r in scores:
            if int(r["k"]) != 1:
                continue
            q = float(r["q_used"])
            by_m[r["method"]][0] += int(float(r["score"]) > q)
            by_m[r["method"]][1] += 1
        out["k1_miscoverage"] = {m: {"miss": a, "n": n, "rate": wilson(a, n)} for m, (a, n) in by_m.items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--scores", default="")
    ap.add_argument("--status", default="")
    ap.add_argument("--robot-id", required=True)
    ap.add_argument("--L", type=float, default=0.6)
    a = ap.parse_args()
    res = analyze(read(a.tracks), read(a.scores) if a.scores else [],
                  read(a.status) if a.status else [], a.robot_id, a.L)
    for k, v in res.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
