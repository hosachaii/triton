# Triton GF(2) Swizzle Implementation Summary

## Overview
This report consolidates the work done to implement and validate the GF(2) swizzle strategy for shared memory in Triton. The goal of this swizzle strategy is to reduce bank conflicts for specific access patterns by replacing the standard XOR swizzle with a GF(2) multiplication-based swizzle for rank-2 tensors not consumed by dot operations.

## Work Accomplished by Stage

### Stage 1: Preparation & Modeling
- Implemented a bank-conflict modeling script (`scratch/model.cpp`) to calculate bank conflicts and wavefronts for local load operations given the expected memory access pattern.
- Validated that for transposed block layouts (where the contiguous thread dimension differs from the contiguous memory dimension), the GF(2) swizzle behaves identically to XOR swizzle in terms of bank conflicts because of the limited range of distinct chunks accessed by a single warp.

### Stage 2: Extending `SharedEncodingAttr`
- Added the `swizzleKind` property to the `SharedEncodingAttr` definition in `TritonGPUAttrDefs.td`. `swizzleKind = 0` denotes the default XOR swizzle, and `swizzleKind = 1` denotes the GF(2) swizzle.
- Updated the MLIR parsers and builders for `SharedEncodingAttr` across the Triton codebase (e.g. `Dialect.cpp`) to support serialization and instantiation of the new `swizzleKind` parameter.

### Stage 3: LLVM Lowering & Integration
- Modified the LLVM lowering phase, specifically `LinearLayoutConversions.cpp`, to utilize the GF(2) swizzle derivation when `swizzleKind == 1` and the tensor properties match the expected criteria (`rank == 2`, `perPhase == 1`).
- Created a manual testing script (`gen_stage3.py`) to generate TTGIR with explicitly set `swizzleKind` and profiled it with NCU via `run_all.py` to observe conflict reductions.

### Stage 4: Automatic Selection
- Implemented auto-selection in `TritonGPUAttrDefs.td` to automatically set `swizzleKind = 1` when the environment variable `TRITON_SHARED_SWIZZLE=auto` is set.
- Ensured this logic only applies to non-dot buffers by injecting it strictly into the `SharedEncodingAttr` builders that do *not* accept `DotOperandEncodingAttr`.

## Key Learnings & Gotchas
- **OOM during builds:** Rebuilding the Triton python package with large job counts resulted in an OOM kill. We learned to enforce `MAX_JOBS=2` via environment variables to stabilize the build.
- **Access Patterns:** Analysis of `local_load` NCU metrics highlighted that for certain access patterns (e.g., transposed layout loads where a warp accesses 32 elements along the strided dimension), the GF(2) swizzle will provide the same bank distribution as the default XOR swizzle because the chunks span only a small domain per warp.

## Next Steps
The GF(2) swizzle is fully implemented in the compiler backend and can be conditionally tested. Further tuning and benchmarking could be performed on kernels with complex block interactions where GF(2) differs strictly from XOR swizzles.
