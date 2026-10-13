# Stage 8: Resolution of Stage 7 Discrepancies

## Executive Summary (<= 15 lines)
```
1. Item 1 (CONFIRMED): LDSM bypasses LSU shared load counters. The 64 load conflicts in P1 MMAv2 arise 100% from the 16 LDS.U.128 scratch reloads in convert_layout (0 in control kernel).
2. Item 2 (REFUTED): The "57% speedup" was cold-start benchmark noise. Stock and p2_m4 have identical PTX/SASS; rigorous interleaved timing shows a 1.64% spread (below 3% harness resolution).
3. Item 3 (REFUTED/EXPLAINED): The 61 cost model mismatches stem entirely from vector-width divergence (shared.getVec() vs emitted LDS/STS width). When widths match, model accuracy is 139/139 (100%).
4. Item 4 (RESOLVED): Part B passed k=1 with maxPhase=4, deactivating GF(2) (nChunks=8 requires maxPhase>=8). Using bank-width k=3 and mp=8 gives 0 conflicts on P1, but plain XOR ties with half the address ops.
5. Item 5 (CONFIRMED): Wavefront doubling on fp32 vec=1 is an access-width fracture (LDS.128 fractured into 4 scalar LDS.32). Matching vec=4 to sizePerThread restores LDS.128, giving 0 conflicts.
6. Item 6 (CONFIRMED): In hypothetical swizzle-aware FMA, plain XOR (vec=1, p=1, m=32) eliminates all conflicts (0.0 conf, 1.0 wf) with 2 ops. GF(2) ties at 0.0 conf but costs 13-19 ops; GF(2) beats XOR on 0 cases.
7. Item 7 (CONFIRMED): Committed predictions for ldmatrix conflicts (0, 32, 96) matched NCU LSU shared pipe totals exactly (0, 32, 96), proving the 4-matrix-group simulation rule is 100% accurate.
8. Bottom Line: GF(2) does not beat tuned plain XOR in any tested or simulated configuration on sm_75.
```

---

## Item 1: Is the P1 Conflict Count from the convert_layout Epilogue?
- **Verdict**: **CONFIRMED**
- **Evidence**:
  - SASS of P1 stock: [`scratch/stage8/p1_stock.sass`](file:///home/hosachai/triton/scratch/stage8/p1_stock.sass#L314-L324) and [`scratch/stage8/p1_stock.sass:498-500`](file:///home/hosachai/triton/scratch/stage8/p1_stock.sass#L498-L500).
    * Main loop dot-operand loads: Exactly 6 `LDSM` instructions (`LDSM.16.M88.4` and `LDSM.16.MT88.4`), 4 `STS.128` operand stores, 16 `HMMA.1688.F32`. Zero `LDS` instructions in the loop body.
    * Epilogue: 8 scalar `STS` stores into scratch buffer, and exactly 2 `LDS.U.128` reload instructions from scratch.
    * Over the grid of 4 CTAs (2 warps/CTA = 8 warps total): $2 \times 8 = 16$ `LDS` instructions executed.
  - Control Kernel: [`scratch/stage8/ctrl_no_convert.ttgir`](file:///home/hosachai/triton/scratch/stage8/ctrl_no_convert.ttgir#L1-L86), disassembled in [`scratch/stage8/ctrl_no_convert.sass`](file:///home/hosachai/triton/scratch/stage8/ctrl_no_convert.sass#L1-L220). Epilogue stores accumulator directly to global memory in `#mma` layout, completely bypassing `convert_layout` scratch allocation.
  - Profiling Data: [`scratch/stage8/ncu_item1.csv`](file:///home/hosachai/triton/scratch/stage8/ncu_item1.csv):
    * P1 Stock (Kernel 2): `smsp__inst_executed_op_shared_ld` = 16, `l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld` = 128 (8.0 wf/inst), `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld` = 64 (4.0 conf/inst).
    * Control Kernel (Kernel 3): `smsp__inst_executed_op_shared_ld` = **0**, `l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld` = **0**, `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld` = **0**.
- **Conclusion**: `LDSM` (`ldmatrix`) does **not** appear in `smsp__inst_executed_op_shared_ld` or `pipe_lsu_mem_shared_op_ld`. The 64 load conflicts in stock MMAv2 kernels are 100% caused by the 16 `LDS.U.128` instructions in the `convert_layout` epilogue.

---

## Item 2: Timing Noise and Retraction of the "57% Speedup" Claim
- **Verdict**: **REFUTED** (Stage 7 speedup claim retracted as benchmark cold-start noise)
- **Evidence**:
  - PTX / SASS Identity: [`scratch/stage7/kernels_partC/p1_stock.ptx`](file:///home/hosachai/triton/scratch/stage7/kernels_partC/p1_stock.ptx) and [`scratch/stage7/kernels_partC/p1_p2_m4.ptx`](file:///home/hosachai/triton/scratch/stage7/kernels_partC/p1_p2_m4.ptx) have **zero diff** (byte-for-byte identical). Both encode `#triton_gpu.shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0]}>`.
  - Rigorous Timing Harness: [`scratch/stage8/timing_harness.py`](file:///home/hosachai/triton/scratch/stage8/timing_harness.py).
    * Protocol: 50 warmup iterations per kernel, 200 timed iterations per batch, 5 randomized interleaved rounds (1,000 timed launches total per candidate), `torch.cuda.synchronize()` around each batch.
  - Raw Timing Results: [`scratch/stage8/timing_results.json`](file:///home/hosachai/triton/scratch/stage8/timing_results.json):
    * P1 Control Pair (`stock` vs `p2_m4`):
      - Stock: Median = 28,368.0 ns, Min = 27,872.0 ns, IQR = 416.0 ns.
      - Variant p2_m4: Median = 27,904.0 ns, Min = 27,712.0 ns, IQR = 320.0 ns.
      - Difference: **1.64%** (well within the < 3.0% harness noise resolution threshold).
    * Part C P1 Candidates Spread: Medians across all valid candidates range from 27,904 ns (`p2_m4`) to 28,528 ns (`p2_m8`)—a total spread of only 2.2%.
    * P2 Control Pair (`stock` vs `p1_m8`):
      - Stock: Median = 36,864.0 ns, Min = 36,640.0 ns, IQR = 384.0 ns.
      - Variant p1_m8: Median = 37,184.0 ns, Min = 36,768.0 ns, IQR = 416.0 ns.
      - Difference: **0.87%** (< 3.0% threshold).
- **Conclusion**: The Stage 7 "57% speedup" (37.2 µs -> 16.4 µs) was pure cold-cache first-run noise in the Stage 7 runner script. The speedup claim is **retracted**.

---

## Item 3: Breakdown of Cost Model Mismatches (61 / 200 Misses)
- **Verdict**: **REFUTED / FULLY EXPLAINED** (Discrepancy is an access-width model mismatch, not cross-warp interference)
- **Evidence**: [`scratch/stage7/part_a_comparison.json`](file:///home/hosachai/triton/scratch/stage7/part_a_comparison.json) and [`scratch/stage8/item3_breakdown.json`](file:///home/hosachai/triton/scratch/stage8/item3_breakdown.json).

### Mismatch Breakdown Table
| Cluster | Count | Dtype | Shape | num_warps | Shared Vec | Emitted SASS | Pred Inst | Meas Inst | Pred WF | Meas WF | Pred Conf | Meas Conf | Root Cause |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **P1_nw8** | 20 | fp16 | 32x32 | nw=8 | 8 | `STS.64` / `LDS.U.64` (8B) | 8 | 8 | 32 | 16 | 0 | 0 | Each thread owns only 4 elements (8 B). SASS emits 64-bit access (2 wf/inst). Model used `vec=8` (16 B, 4 wf/inst), over-predicting wavefronts by $2\times$. |
| **P3 (col)** | 24 | fp32 | 32x32, 64x32 | nw=2,4,8 | 1 | `STS.128` / `LDS.U.128` (16B) | 32, 64 | 8, 16 | 128, 256 | 32, 64 | 96, 192 | 0 | Contiguous blocked layout enabled 128-bit vectorization (`.v4.u32`). Model assumed `vec=1` scalar loads (32 lanes/group conflicting). Vector loads execute in 8-lane quarter-warps with **0 conflicts**. |
| **P4 (row)** | 16 | fp32 | 32x32, 64x32 | nw=2,4,8 | 1 | `STS.128` / `LDS.U.128` (16B) | 32, 64 | 8, 16 | 128, 256 | 32, 64 | 96, 192 | 0 | Same as P3: LLVM vectorized to `LDS.128` (vec=4, 0 conflicts). Model assumed scalar 32-bit accesses (`vec=1`). |
| **P5_32x64** | 1 | int8 | 32x64 | nw=4 | 8 | `STS.32` / `LDS.U.128` (16B) | 8 | 4 | 32 | 16 | 16 | 0 | Unswizzled linear layout allowed LLVM to emit `LDS.128` (16B, vec=16), halving instruction count vs model assumption. |

- **Investigation of Hypotheses**:
  - "Cross-warp interference": **REFUTED**. Warps issue shared memory instructions independently; hardware LSU banks do not produce inter-warp bank conflicts within a memory transaction.
  - Counting consistency: NCU counts total wavefronts/conflicts per operation type across all CTA warps. The cost model sums across all warps and instructions.
- **Reliability Conditions**:
  - The model is **100% reliable (139 / 139 exact matches)** if and only if:
    1. The shared encoding `shared.getVec()` matches the vector width emitted by the compiler (i.e., `vec == emitted_vec`).
    2. Number of elements per thread is $\ge$ `shared.getVec()`.
  - When `emitted_vec != shared.getVec()`, accuracy is 0% (61 / 61 misses).
  - To fix the model: Replace `vec = shared.getVec()` in [`LinearLayoutConversions.cpp:1217`](file:///home/hosachai/triton/lib/Dialect/TritonGPU/IR/LinearLayoutConversions.cpp#L1217) with `effectiveVec = min(shared.getVec(), contiguousElementsPerThread, 16 / elemBytes)`.

---

## Item 4: GF(2) Parameter Calculations and C++ Implementation Verification
- **Verdict**: **RESOLVED**
- **Evidence**: [`scratch/stage8/item4_results.json`](file:///home/hosachai/triton/scratch/stage8/item4_results.json) and [`lib/Dialect/TritonGPU/IR/LinearLayoutConversions.cpp:371-381`](file:///home/hosachai/triton/lib/Dialect/TritonGPU/IR/LinearLayoutConversions.cpp#L371-L381).

### Full Parameter Computations
- **Implementation Rule**: The C++ code strictly uses the **128 B bank-width chunk count**:
  `nChunks = 128 / (vec * elemBytes)`, `k = Log2(nChunks)`, with condition `maxPhase >= nChunks` and $2 \le k \le 7$.
  It does **not** use the row-limited chunk count ($64\text{ B} / 16\text{ B} = 4$).
- **Part A vs Part B Reconciliation**:
  * Part A: Used `gf2_p1_m8` (`maxPhase=8`). For fp16 `vec=8`: `nChunks = 128 / 16 = 8`. Since $maxPhase \ge nChunks$, GF(2) was **active** ($k=3$).
  * Part B: Listed `maxPhase=4, k=1`. In `simulate_banks.py`, line 98 checked `maxPhase >= nChunks`. Because $4 \ge 8$ is **false**, GF(2) was **deactivated** and executed plain XOR `p=1, m=4` (giving 4 conflicts).
- **Part B Simulation with All Plausible $k$**:
  * $k=1$ ($mp=4$ or $mp=8$): 16 wf / 12 conflicts per instruction.
  * $k=2$ (row-limited, $mp=4$): 8 wf / 4 conflicts per instruction.
  * $k=3$ (bank-width, $mp=4$): 8 wf / 4 conflicts per instruction (modulo row wrap collides row 0 and 2).
  * $k=3$ (bank-width, $mp=8$): **4 wf / 0 conflicts** per instruction.
- **Does "GF(2) is worse than plain XOR on P1" change?**:
  * **No**. Tuned stock plain XOR achieves **0 conflicts** with `maxPhase=4` and only 7 PTX address ops. GF(2) requires `maxPhase=8` and 14–19 address ops to achieve 0 conflicts. GF(2) never beats plain XOR on P1.
- **Agreement with C++ Implementation**:
  * Simulated [`scratch/gf2_reference.py:gf2_swizzle`](file:///home/hosachai/triton/scratch/gf2_reference.py#L45-L50) tested against the $\mathbb{F}_2$ basis vectors from [`LinearLayoutConversions.cpp:387-408`](file:///home/hosachai/triton/lib/Dialect/TritonGPU/IR/LinearLayoutConversions.cpp#L387-L408) across all $(row, chunk)$ inputs for $k \in [2..7]$:
  * **21,840 pairs tested, exactly 0 mismatches**.

---

## Item 5: Wavefront Doubling on fp32 vec=1 (P3, P4, P5_32x64)
- **Verdict**: **CONFIRMED** (Access-width fracture, not a bank-mapping effect)
- **Evidence**:
  - PTX / SASS Analysis:
    * P3 Stock (`k_pA_090`): [`scratch/stage8/k_pA_090.cubin`](file:///home/hosachai/triton/scratch/stage8/k_pA_090.cubin). SASS shows 2 `LDS.U.128` (16 B) and 2 `STS.128` instructions per thread (8 instructions executed across CTA). Wavefronts = 32, Conflicts = 0.
    * P3 GF(2) at `vec=1` (`k_pA_099`): [`scratch/stage8/k_pA_099.cubin`](file:///home/hosachai/triton/scratch/stage8/k_pA_099.cubin). SASS shows 8 `LDS.U` (4 B scalar) and 8 `STS` instructions per thread (32 instructions executed across CTA). Wavefronts = 64, Conflicts = 32.
    * P4 Stock (`k_pA_150`): SASS shows 2 `LDS.U.128` / 2 `STS.128` (0 conflicts).
    * P4 GF(2) at `vec=1` (`k_pA_159`): SASS shows 8 `LDS.U` / 8 `STS` (32 conflicts).
  - Hypothesis Test (Matching `vec` to `sizePerThread`):
    * Built and compiled P3 with `shared_vec=4, swizzleKind=1` ($k=3$): [`scratch/stage8/test_p3_vec4_gf2.ttgir`](file:///home/hosachai/triton/scratch/stage8/test_p3_vec4_gf2.ttgir).
    * SASS: Restored exactly 2 `LDS.U.128` and 2 `STS.128` instructions per thread.
    * NCU Profile ([`scratch/stage8/ncu_p3_vec4.csv`](file:///home/hosachai/triton/scratch/stage8/ncu_p3_vec4.csv)): `inst_ld` = 8, `ld_wavefronts` = 32, `ld_conflicts` = **0**.
  - P5_32x64 Store Analysis:
    * Both stock (`k_pA_190`) and GF(2) (`k_pA_199`) emit `STS.64`.
    * GF(2) with $k=4$ (`maxPhase=16`) wraps modulo 64 columns, collapsing row bit 3 and causing rows 0 and 8 to collide on shared banks (16 conflicts).

---

## Item 6: Swizzle-Aware FMA Loader Simulation
- **Verdict**: **CONFIRMED** (Hypothetical loader simulation; GF(2) never beats plain XOR)
- **Status Label**: **Simulated (Hypothetical loader; does not exist in Triton source today)**
- **Evidence**: [`scratch/stage8/simulate_fma_swizzle.py`](file:///home/hosachai/triton/scratch/stage8/simulate_fma_swizzle.py) and [`scratch/stage8/item6_fma_simulation.json`](file:///home/hosachai/triton/scratch/stage8/item6_fma_simulation.json).
- **Validation**:
  * P3 Stock (fp32 col-major B, unswizzled): 32 threads map to 16 columns (2 threads per column hit identical addresses $\to$ hardware broadcast). Exactly 16 distinct words hit 1 bank $\to$ 16.0 wf/inst, 15.0 conf/inst (93.8% fraction on B-loader). Combined with 0-conflict A-loader and CTA stores, whole-kernel conflict fraction is 81.8% (reproducing Stage 6 row 183).
  * P4 Stock (fp32 row-major B, unswizzled): 1.0 wf/inst, 0.0 conflicts (100% exact match).

### Sweep Results (Best Plain XOR vs Best GF(2))
| Pattern | Best Plain XOR | XOR Wf (Conf) | XOR Ops | Best GF(2) | GF(2) Wf (Conf) | GF(2) Ops | GF(2) Beats Plain XOR? |
|---|---|---|---|---|---|---|---|
| **fp32 col 32x32** | vec=1, p=1, m=32 | 1.0 (0.0) | 2 | vec=1, k=5 | 1.0 (0.0) | 19 | **NO (Tied on conflicts, worse on ops)** |
| **fp32 col 64x32** | vec=1, p=1, m=32 | 1.0 (0.0) | 2 | vec=1, k=5 | 1.0 (0.0) | 19 | **NO (Tied on conflicts, worse on ops)** |
| **fp32 row 32x32** | vec=1, p=1, m=1 | 1.0 (0.0) | 1 | vec=1, k=5 | 1.0 (0.0) | 13 | **NO (Tied on conflicts, worse on ops)** |
| **fp32 row 64x32** | vec=1, p=1, m=1 | 1.0 (0.0) | 1 | vec=1, k=5 | 1.0 (0.0) | 13 | **NO (Tied on conflicts, worse on ops)** |
| **fp16 col 32x32** | vec=1, p=1, m=32 | 1.0 (0.0) | 2 | vec=1, k=5 | 1.0 (0.0) | 19 | **NO (Tied on conflicts, worse on ops)** |
| **fp16 col 64x32** | vec=1, p=1, m=32 | 1.0 (0.0) | 2 | vec=1, k=5 | 1.0 (0.0) | 19 | **NO (Tied on conflicts, worse on ops)** |
| **fp16 row 32x32** | vec=1, p=1, m=1 | 1.0 (0.0) | 1 | vec=1, k=5 | 1.0 (0.0) | 13 | **NO (Tied on conflicts, worse on ops)** |
| **fp16 row 64x32** | vec=1, p=1, m=1 | 1.0 (0.0) | 1 | vec=1, k=5 | 1.0 (0.0) | 13 | **NO (Tied on conflicts, worse on ops)** |
| **int8 col 32x32** | vec=1, p=1, m=32 | 1.0 (0.0) | 2 | vec=1, k=5 | 1.0 (0.0) | 19 | **NO (Tied on conflicts, worse on ops)** |
| **int8 col 64x32** | vec=2, p=1, m=32 | 1.0 (0.0) | 5 | vec=1, k=5 | 1.0 (0.0) | 19 | **NO (Tied on conflicts, worse on ops)** |
| **int8 row 32x32** | vec=1, p=1, m=1 | 1.0 (0.0) | 1 | vec=1, k=5 | 1.0 (0.0) | 13 | **NO (Tied on conflicts, worse on ops)** |
| **int8 row 64x32** | vec=1, p=1, m=1 | 1.0 (0.0) | 1 | vec=1, k=5 | 1.0 (0.0) | 13 | **NO (Tied on conflicts, worse on ops)** |

- **Conclusion**: There is **no pattern where GF(2) beats plain XOR**. Plain XOR achieves the theoretical lower bound (0.0 conflicts, 1.0 wf) across all patterns with 1–5 address ops, whereas GF(2) requires 13–19 ops.

---

## Item 7: Validation of the 4-Matrix-Group ldmatrix Simulation Rule
- **Verdict**: **CONFIRMED** (Exact 100% agreement between committed predictions and NCU measurement)
- **Predictions Pre-Commit**: [`scratch/stage8/predictions_item7.txt`](file:///home/hosachai/triton/scratch/stage8/predictions_item7.txt) committed to git at `a3bf3133bd` prior to measurement.
- **Experimental Setup**:
  * Evaluated 3 control MMAv2 kernels with no epilogue conversion on 1 CTA (8 `LDSM` instructions per CTA):
    1. `ctrl_stock`: `vec=8, perPhase=2, maxPhase=4` (predicts 0 conf/inst $\to$ 0 total)
    2. `ctrl_p1_m4`: `vec=8, perPhase=1, maxPhase=4` (predicts 4 conf/inst $\to$ 32 total)
    3. `ctrl_p1_m2`: `vec=8, perPhase=1, maxPhase=2` (predicts 12 conf/inst $\to$ 96 total)
- **Measurement Results**: [`scratch/stage8/ncu_item7.csv`](file:///home/hosachai/triton/scratch/stage8/ncu_item7.csv) and [`scratch/stage8/item7_results.json`](file:///home/hosachai/triton/scratch/stage8/item7_results.json):
  * Profiler counter: As established in Item 1, `LDSM` does not increment `op_ld` sub-counters. However, `LDSM` transactions are fully serviced by the LSU shared memory crossbar and captured by `l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum`.
  * Measured vs Predicted:
    - `ctrl_stock`: Predicted = **0**, Measured = **0.0** (Exact match)
    - `ctrl_p1_m4`: Predicted = **32**, Measured = **32.0** (Exact match)
    - `ctrl_p1_m2`: Predicted = **96**, Measured = **96.0** (Exact match)
  * Wavefronts: Measured total shared pipe wavefronts were 68.0, 100.0, and 164.0 ($+32$ and $+96$ over stock).
- **Conclusion**: The 4-matrix-group simulation rule for `ldmatrix` is **100% correct**.

---

## Item 8: Final Consolidated Statements

### Table of Stage 7 Claims and Stage 8 Verdicts
| # | Stage 7 Claim (Executive Summary) | Stage 8 Verdict | Evidence File |
|---|---|---|---|
| 1 | Cost Model: 69.5% match; worst miss P3_64x32_nw2 | **REFUTED / EXPLAINED** | [`scratch/stage8/item3_breakdown.json`](file:///home/hosachai/triton/scratch/stage8/item3_breakdown.json) |
| 2 | Part A GF(2) vs Best XOR: GF(2) tied or strictly worse on conflicts (+extra ops) | **CONFIRMED** | [`scratch/stage7/part_a_comparison.json`](file:///home/hosachai/triton/scratch/stage7/part_a_comparison.json), [`scratch/stage8/test_p3_vec4_gf2.ptx`](file:///home/hosachai/triton/scratch/stage8/test_p3_vec4_gf2.ptx) |
| 3 | Part B Simulation Validation: P3 96.9% sim vs 81.8% meas matched | **REFUTED** | [`scratch/stage8/simulate_fma_swizzle.py`](file:///home/hosachai/triton/scratch/stage8/simulate_fma_swizzle.py) |
| 4 | Part B MMAv2 P1: 64 measured conflicts arose from epilogue scratch reloads | **CONFIRMED** | [`scratch/stage8/p1_stock.sass`](file:///home/hosachai/triton/scratch/stage8/p1_stock.sass), [`scratch/stage8/ncu_item1.csv`](file:///home/hosachai/triton/scratch/stage8/ncu_item1.csv) |
| 5 | Part B Tuning & GF(2): GF(2) worse on P1 (4 conf); P3 both 32 wf/inst | **REFUTED** | [`scratch/stage8/item4_results.json`](file:///home/hosachai/triton/scratch/stage8/item4_results.json), [`scratch/stage8/item6_fma_simulation.json`](file:///home/hosachai/triton/scratch/stage8/item6_fma_simulation.json) |
| 6 | Part C MMAv2: Changing dot #shared does not remove epilogue scratch conflicts | **CONFIRMED** | [`scratch/stage8/ctrl_no_convert.sass`](file:///home/hosachai/triton/scratch/stage8/ctrl_no_convert.sass), [`scratch/stage8/ncu_item1.csv`](file:///home/hosachai/triton/scratch/stage8/ncu_item1.csv) |
| 7 | Part C Timing: Variants p2_m2 and p1_m4 gave 57% speedup | **REFUTED** | [`scratch/stage8/timing_results.json`](file:///home/hosachai/triton/scratch/stage8/timing_results.json) |
| 8 | Part C Agreement: Measured ldmatrix agrees with Part B simulation (0 conflicts) | **CONFIRMED** | [`scratch/stage8/ncu_item7.csv`](file:///home/hosachai/triton/scratch/stage8/ncu_item7.csv), [`scratch/stage8/item7_results.json`](file:///home/hosachai/triton/scratch/stage8/item7_results.json) |
| 9 | Verifier & Loader Constraints: GF(2) dot rejected by verifier; MMAv2 ignores it | **CONFIRMED** | [`lib/Dialect/TritonGPU/IR/Dialect.cpp:3060`](file:///home/hosachai/triton/lib/Dialect/TritonGPU/IR/Dialect.cpp#L3060) |
| 10 | FMA Loader Breakage: SharedToDotOperandFMA uses naive linear offsets | **CONFIRMED** | [`lib/Conversion/TritonGPUToLLVM/ConvertLayoutOpToLLVM/SharedToDotOperandFMA.cpp:144-218`](file:///home/hosachai/triton/lib/Conversion/TritonGPUToLLVM/ConvertLayoutOpToLLVM/SharedToDotOperandFMA.cpp#L144-L218) |
| 11 | Untested Scopes: Hardware loaders, swizzled FMA, Ampere/Hopper MMA | **CONFIRMED** | Documented in Section 8 conclusions below |
| 12 | Environment: GTX 1650 sm_75 unlocked clocks; 20-run timings indicative only | **CONFIRMED** | [`scratch/stage8/timing_harness.py`](file:///home/hosachai/triton/scratch/stage8/timing_harness.py) |

### Concluding Questions (<= 8 lines)
```
1. Is there any tested or simulated case where GF(2) beats the best tuned plain XOR on conflicts? None.
2. Is there any case where GF(2) would plausibly beat it but loader/consumer does not exist? None; plain XOR achieves 0.0 conflicts across all simulated FMA patterns with fewer address ops.
3. Is the cost model reliable enough to drive an automatic selector? Yes, conditionally: 100% reliable when shared.getVec() equals the compiler's emitted vector width, but unreliable when LLVM widens or fractures accesses.
4. What remains untested? Hardware-level swizzled FMA lowering in LLIR, Ampere/Hopper MMA architectures, and non-2D/reduction shared memory scratch layouts.
```
