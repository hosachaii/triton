#!/usr/bin/env python3
"""
Rigorous Timing Harness for Stage 8 Item 2.
- >= 50 warmup launches per kernel
- >= 200 timed launches per kernel
- Randomised interleaved ordering across 5 rounds (40 launches per round = 200 total)
- torch.cuda.synchronize around each launch batch
- Reports median, IQR (Q3 - Q1), min for each candidate
- Computes difference between identical-encoding control pairs:
    P1: stock vs p2_m4 (both are p=2, m=4)
    P2: stock vs p1_m8 (both are p=1, m=8)
"""

import os
import sys
import json
import random
import numpy as np
import torch
import triton

os.environ["TRITON_ALWAYS_COMPILE"] = "1"

def benchmark_suite(items, pattern, num_rounds=5, launches_per_round=40, warmup_count=50):
    print(f"\n=======================================================")
    print(f"BENCHMARKING {pattern} VARIANTS (Total timed runs per kernel: {num_rounds * launches_per_round})")
    print(f"=======================================================")

    M, N, K = 64, 64, 64
    grid = (2, 2, 1) if pattern == "P1" else (1, 1, 1)

    a = torch.randn((M, K), dtype=torch.float16, device="cuda")
    b = torch.randn((K, N), dtype=torch.float16, device="cuda")
    c = torch.empty((M, N), dtype=torch.float16, device="cuda")

    # Compile all kernels
    compiled_kernels = {}
    for it in items:
        k = triton.compile(it["path"])
        compiled_kernels[it["name"]] = k

    # Warmup all kernels >= 50 times
    print(f"Running {warmup_count} warmup launches per kernel...")
    for name, k in compiled_kernels.items():
        for _ in range(warmup_count):
            k[grid](a, b, c, M, N, K, K, N, N)
    torch.cuda.synchronize()

    # Data collection: list of times (in ns) per kernel
    results_times = {it["name"]: [] for it in items}

    # Interleaved rounds
    candidate_names = [it["name"] for it in items]

    for round_idx in range(num_rounds):
        # Randomise candidate order for this round
        order = candidate_names.copy()
        random.shuffle(order)
        print(f"  Round {round_idx + 1}/{num_rounds} (order: {order[:3]}...)")

        for name in order:
            k = compiled_kernels[name]
            torch.cuda.synchronize()
            
            # Record events for each launch in this round
            starts = [torch.cuda.Event(enable_timing=True) for _ in range(launches_per_round)]
            ends = [torch.cuda.Event(enable_timing=True) for _ in range(launches_per_round)]

            for i in range(launches_per_round):
                starts[i].record()
                k[grid](a, b, c, M, N, K, K, N, N)
                ends[i].record()

            torch.cuda.synchronize()

            for i in range(launches_per_round):
                t_ns = starts[i].elapsed_time(ends[i]) * 1e6
                results_times[name].append(t_ns)

    # Compute statistics
    stats = {}
    for it in items:
        name = it["name"]
        arr = np.array(results_times[name])
        med = float(np.median(arr))
        q25 = float(np.percentile(arr, 25))
        q75 = float(np.percentile(arr, 75))
        iqr = float(q75 - q25)
        min_v = float(np.min(arr))
        mean_v = float(np.mean(arr))
        std_v = float(np.std(arr))

        stats[name] = {
            "name": name,
            "perPhase": it["perPhase"],
            "maxPhase": it["maxPhase"],
            "median_ns": med,
            "iqr_ns": iqr,
            "min_ns": min_v,
            "mean_ns": mean_v,
            "std_ns": std_v,
            "n_runs": len(arr),
            "raw_times": arr.tolist()
        }

    return stats

def main():
    with open("scratch/stage7/part_c_comparison.json") as f:
        part_c_meta = json.load(f)

    # Filter valid P1 and P2 items
    p1_items = [
        {"name": "stock", "perPhase": 2, "maxPhase": 4, "path": "scratch/stage7/kernels_partC/p1_stock.ttgir"},
        {"name": "p1_m1", "perPhase": 1, "maxPhase": 1, "path": "scratch/stage7/kernels_partC/p1_p1_m1.ttgir"},
        {"name": "p1_m2", "perPhase": 1, "maxPhase": 2, "path": "scratch/stage7/kernels_partC/p1_p1_m2.ttgir"},
        {"name": "p1_m4", "perPhase": 1, "maxPhase": 4, "path": "scratch/stage7/kernels_partC/p1_p1_m4.ttgir"},
        {"name": "p2_m2", "perPhase": 2, "maxPhase": 2, "path": "scratch/stage7/kernels_partC/p1_p2_m2.ttgir"},
        {"name": "p2_m4", "perPhase": 2, "maxPhase": 4, "path": "scratch/stage7/kernels_partC/p1_p2_m4.ttgir"}, # Control pair with stock!
        {"name": "p2_m8", "perPhase": 2, "maxPhase": 8, "path": "scratch/stage7/kernels_partC/p1_p2_m8.ttgir"},
    ]

    p2_items = [
        {"name": "stock", "perPhase": 1, "maxPhase": 8, "path": "scratch/stage7/kernels_partC/p2_stock.ttgir"},
        {"name": "p1_m1", "perPhase": 1, "maxPhase": 1, "path": "scratch/stage7/kernels_partC/p2_p1_m1.ttgir"},
        {"name": "p1_m2", "perPhase": 1, "maxPhase": 2, "path": "scratch/stage7/kernels_partC/p2_p1_m2.ttgir"},
        {"name": "p1_m4", "perPhase": 1, "maxPhase": 4, "path": "scratch/stage7/kernels_partC/p2_p1_m4.ttgir"},
        {"name": "p1_m8", "perPhase": 1, "maxPhase": 8, "path": "scratch/stage7/kernels_partC/p2_p1_m8.ttgir"}, # Control pair with stock!
        {"name": "p2_m2", "perPhase": 2, "maxPhase": 2, "path": "scratch/stage7/kernels_partC/p2_p2_m2.ttgir"},
        {"name": "p2_m4", "perPhase": 2, "maxPhase": 4, "path": "scratch/stage7/kernels_partC/p2_p2_m4.ttgir"},
        {"name": "p1_m16", "perPhase": 1, "maxPhase": 16, "path": "scratch/stage7/kernels_partC/p2_p1_m16.ttgir"},
    ]

    p1_stats = benchmark_suite(p1_items, "P1")
    p2_stats = benchmark_suite(p2_items, "P2")

    # Control pair check for P1
    stock_p1 = p1_stats["stock"]
    ctrl_p1 = p1_stats["p2_m4"]
    diff_p1 = abs(stock_p1["median_ns"] - ctrl_p1["median_ns"]) / stock_p1["median_ns"] * 100.0

    print("\n=======================================================")
    print("IDENTICAL ENCODING CONTROL PAIR CHECK:")
    print("=======================================================")
    print(f"P1 Control Pair: 'stock' vs 'p2_m4' (both are perPhase=2, maxPhase=4):")
    print(f"  stock: median = {stock_p1['median_ns']:.1f} ns, IQR = {stock_p1['iqr_ns']:.1f} ns, min = {stock_p1['min_ns']:.1f} ns")
    print(f"  p2_m4: median = {ctrl_p1['median_ns']:.1f} ns, IQR = {ctrl_p1['iqr_ns']:.1f} ns, min = {ctrl_p1['min_ns']:.1f} ns")
    print(f"  Relative difference: {diff_p1:.2f}% (Threshold: 3.0%)")

    stock_p2 = p2_stats["stock"]
    ctrl_p2 = p2_stats["p1_m8"]
    diff_p2 = abs(stock_p2["median_ns"] - ctrl_p2["median_ns"]) / stock_p2["median_ns"] * 100.0

    print(f"\nP2 Control Pair: 'stock' vs 'p1_m8' (both are perPhase=1, maxPhase=8):")
    print(f"  stock: median = {stock_p2['median_ns']:.1f} ns, IQR = {stock_p2['iqr_ns']:.1f} ns, min = {stock_p2['min_ns']:.1f} ns")
    print(f"  p1_m8: median = {ctrl_p2['median_ns']:.1f} ns, IQR = {ctrl_p2['iqr_ns']:.1f} ns, min = {ctrl_p2['min_ns']:.1f} ns")
    print(f"  Relative difference: {diff_p2:.2f}% (Threshold: 3.0%)")

    output_data = {
        "P1_control_diff_pct": diff_p1,
        "P2_control_diff_pct": diff_p2,
        "P1_stats": {k: {kk: vv for kk, vv in v.items() if kk != "raw_times"} for k, v in p1_stats.items()},
        "P2_stats": {k: {kk: vv for kk, vv in v.items() if kk != "raw_times"} for k, v in p2_stats.items()},
    }

    with open("scratch/stage8/timing_results.json", "w") as f:
        json.dump(output_data, f, indent=2)

    # Print summary table
    print("\n=======================================================")
    print("P1 SUMMARY TABLE (200 RUNS)")
    print("=======================================================")
    print(f"{'Variant':12s} | {'perPhase':8s} | {'maxPhase':8s} | {'Median (ns)':12s} | {'IQR (ns)':10s} | {'Min (ns)':10s}")
    print("-" * 75)
    for k, v in p1_stats.items():
        print(f"{v['name']:12s} | {v['perPhase']:8d} | {v['maxPhase']:8d} | {v['median_ns']:12.1f} | {v['iqr_ns']:10.1f} | {v['min_ns']:10.1f}")

    print("\n=======================================================")
    print("P2 SUMMARY TABLE (200 RUNS)")
    print("=======================================================")
    print(f"{'Variant':12s} | {'perPhase':8s} | {'maxPhase':8s} | {'Median (ns)':12s} | {'IQR (ns)':10s} | {'Min (ns)':10s}")
    print("-" * 75)
    for k, v in p2_stats.items():
        print(f"{v['name']:12s} | {v['perPhase']:8d} | {v['maxPhase']:8d} | {v['median_ns']:12.1f} | {v['iqr_ns']:10.1f} | {v['min_ns']:10.1f}")

if __name__ == "__main__":
    main()
