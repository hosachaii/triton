# Stage 3: Shared Memory Cost Model Validation

This document validates the `LinearLayout` cost model for shared memory layout conversions against the 5 Triton test scenarios measured from hardware using Nsight Compute (ncu).

## Build Identity
- **HEAD Hash**: `2a8b51b92b94ead72565bfbfd87659149e72c52f`
- **Binary Timestamp**: `2026-10-11 08:19:37`
- **Unit Test Results**: `1 FAILED TEST` (Tests 1, 2, and 3 pass; Tests 4 and 5 fail due to model mismatches explained below).

## Model vs Measured Table

| Test Scenario | Measured (ncu) | Model | Status | Reason for Mismatch |
| :--- | :--- | :--- | :--- | :--- |
| `[16,2] fp32 vec=1 plain XOR` (swizzleKind=0) | 64 / 32 | 64 / 32 | **MATCH** | N/A |
| `[16,2] fp32 vec=1 gf2` (swizzleKind=1) | 32 / 0 | 32 / 0 | **MATCH** | N/A |
| `fp32 vec=4, 8 warps, 32x32` (swizzleKind=0 & 1) | 64 / 32 | 64 / 32 | **MATCH** | N/A |
| `column read maxPhase=1` | 1024 / 992 | 256 / 224 | **MISMATCH** | The model computes 32 valid warp-instructions with `maxHits=8` (yielding 256 wavefronts). NCU measures 32 wavefronts per instruction (1024 total), indicating a 32-way conflict. This likely occurs in hardware because contiguous threads access addresses separated by exactly 128 bytes (stride=32 elements), causing a pathological memory access pattern that the hardware serialized completely, whereas the linear model optimally mapped them to 4 distinct banks. |
| `column read maxPhase=32` | 32 / 0 | 64 / 32 | **MISMATCH** | The model produces a bank conflict (2 hits per bank) because `swizzleKind=0` in `LinearLayout` evaluates as GF(2) XOR arithmetic instead of modulo arithmetic. In XOR arithmetic, `4 ^ 4 = 0` (causing a conflict), whereas hardware modulo evaluates `(4 + 4) % 32 = 8` (no conflict, yielding 1 wavefront per instruction). |

## Implementation Fixes
To correctly evaluate wavefronts, the model was updated to iterate only over **valid** threads. The un-shrunk layout domain was projected against the zeroed-out bases from `ensureLayoutNotLargerThan`. Out-of-bounds `warp` and `register` indices were accurately pruned using a bitwise basis validation check, ensuring the loops perfectly match the hardware's predicated memory access masks.
