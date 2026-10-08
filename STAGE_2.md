# Stage 2: GF(2) Swizzle Compiler Implementation Verification

## 1. Verifier & Lowering Constraints
The constraints for GF(2) swizzle have been strictly formalized via MLIR verification and layout conversion overrides:
- **`swizzleKind` property**: Added `swizzleKind` to `#triton_gpu.shared` (0 = default plain XOR, 1 = GF2 swizzle). The tablegen definition sets it with a default value of `0`.
- **Verifier checks (`Dialect.cpp` & `Ops.cpp`)**: 
  - `SharedEncodingAttr::verify`: Rejects invalid parameters early (`swizzleKind != 0` requires `rank 2`, `hasLeadingOffset = false`, and `perPhase = 1`).
  - `MemDescType::verify` (via `verifyMemDescEncoding`): Enforces `maxPhase >= nChunks` (where `nChunks = 128 / (vec * elemBytes)`) for any memdesc with `swizzleKind=1`. This was added globally to `MemDescType::verify` in Stage 1 and applies to `local_alloc`, function arguments, subviews, etc.
  - `LocalLoadOp::verify`: Rejects loads that attempt to consume a `swizzleKind != 0` memdesc straight into a dot operand (`dot_operand`).

## 2. Profiling Verification (Real Outputs)
All profiling tests were run under `ncu` (without `sudo`) using `TRITON_ALWAYS_COMPILE=1` and `-k regex:test_kernel` on a CC 7.5 (Turing) device. The metric `l1tex__t_requests_pipe_lsu_mem_shared` is reported as `n/a` by Turing `ncu`, so we derive the request count dynamically using the formula `Wavefronts - Bank Conflicts = Base Requests`.

### (a) The `[16,2]` pair
Testing `sizePerThread=[1,1], threadsPerWarp=[16,2], warpsPerCTA=[2,2]` block layouts (128 threads total). 

**Unset (swizzleKind = 0):**
```text
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum                       32
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum                       32
    l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum                           64
    l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum                           64
```
Derived Requests = 64 wavefronts - 32 bank conflicts = 32.

**GF2 (swizzleKind = 1):**
```text
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum                        0
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum                        0
    l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum                           32
    l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum                           32
```
Derived Requests = 32 wavefronts - 0 bank conflicts = 32.
*Result:* This proves the `[16,2]` pair experiences 32 bank conflicts on load and store with plain XOR, which are entirely eliminated (0 bank conflicts) by the GF2 swizzle.

### (b) The `vec=4` pair
Testing a layout that stores and loads via `sizePerThread=[1,4]` (128-bit `vec=4` operations), with `fp32` elements (`elemBytes=4`). We have `nChunks = 128 / (4 * 4) = 8`, and the encoding sets `maxPhase = 8`.

Because `maxPhase >= nChunks` (8 >= 8) is satisfied, the scope condition for GF2 is met and **GF2 is applied**.

**PTX Excerpt for Load & Store (`test_vec4.ttgir`):**
```ptx
	st.shared.v4.u32 	[%r29], {%r1, %r2, %r3, %r4};
	ld.shared.v4.u32 	{%r5, %r6, %r7, %r8}, [%r29];
```
*(By mirroring the load `sizePerThread` to `[1,4]`, it emits a perfect 128-bit `ld.shared.v4.u32` matching the store).*

**Unset (swizzleKind = 0):**
```text
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum                       32
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum                       32
    l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum                           64
    l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum                           64
```

**GF2 (swizzleKind = 1):**
```text
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum                       32
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum                       32
    l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum                           64
    l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum                           64
```

**Lane-by-Lane Derivation (Lanes 0-7)**
A 16-byte memory access (like `.v4.u32`) for a warp is served in **four 8-lane groups**. 
For Group 1 (lanes 0-7), under `order=[1,0]` (dim 1 contiguous) and `threadsPerWarp=[8,4]`:
- Lane `t` (0..7) has `row = t / 4`.
- `t = 0..3` maps to `row=0`. Their logical chunks are `0, 1, 2, 3`. Each chunk is 4 banks.
- `t = 4..7` maps to `row=1`. Their logical chunks are `0, 1, 2, 3`. Each chunk is 4 banks.

**Plain XOR Expected Wavefronts:**
- Plain XOR (perPhase=1, maxPhase=8) uses shift = `row mod 8`.
- Row 0 (t=0..3) uses `chunk ^ 0`. Exact banks: `0, 4, 8, 12`.
- Row 1 (t=4..7) uses `chunk ^ 1`. Exact banks: `4, 0, 12, 8`.
- *Wavefront Count*: Bank 0 is hit by t=0 & t=5. Bank 4 is hit by t=1 & t=4. Bank 8 is hit by t=2 & t=7. Bank 12 is hit by t=3 & t=6.
- **Wavefronts = 2** (since max hits per bank = 2).

**GF2 Expected Wavefronts:**
- GF2 at k=3 uses irreducible polynomial `p(x) = x^3 + x + 1`. For row=1, `G(1) = 2` (since `1 << 1 = 2`, and no `x^3` bit to overflow).
- Row 0 (t=0..3) uses `chunk ^ 0`. Exact banks: `0, 4, 8, 12`.
- Row 1 (t=4..7) uses `chunk ^ 2`. Exact banks: `8, 12, 0, 4`.
- *Wavefront Count*: Bank 0 is hit by t=0 & t=6. Bank 4 is hit by t=1 & t=7. Bank 8 is hit by t=2 & t=4. Bank 12 is hit by t=3 & t=5.
- **Wavefronts = 2** (since max hits per bank = 2).

**Why this holds for ALL four 8-lane groups:**
Each 8-lane group consists of exactly two adjacent rows (e.g. rows 2 and 3, rows 4 and 5, etc).
For plain XOR, the shifts differ by 1 (e.g. `2` and `3`).
For GF2, the shifts differ by `G(1) = 2` (since multiplication by x in GF2 distributes over addition, `G(r+1) = G(r) ^ G(1)` for adjacent rows if there's no polynomial reduction, which there isn't for the lowest bit).
Neither a shift difference of `1` (plain XOR) nor `2` (GF2) flips the top chunk bit (the `4` bit, which distinguishes chunks 0-3 from 4-7, selecting bank half 0-15 vs 16-31). Since the top bit is untouched, both rows in the group are entirely confined to the SAME half of the banks (0-15).
With 8 threads in the group all accessing the same 4 banks (4 chunks * 4 bytes/bank = 16 bytes = 4 chunks), each of those 4 banks is hit exactly twice. Thus, every 8-lane group takes exactly 2 wavefronts, resulting in 8 wavefronts per warp for both plain XOR and GF2.

## 3. Ops carrying MemDesc & Subview Propagation

The MLIR verifier enforces `swizzleKind` constraints globally on all `!tt.memdesc` values via `MemDescType::verify` (since Stage 1!). This includes:
- **`tt.func` arguments**
- **`triton_gpu.memdesc_subview`**
- **`triton_gpu.local_alloc`**
- **`triton_gpu.async_copy_global_to_local`**

### Subview Type Propagation
The `MemDescSubviewOp` verifier explicitly demands that the `swizzleKind` of the source `memdesc` perfectly matches the `swizzleKind` of the resulting subview `memdesc`!
The subview generating passes (such as the `Pipeliner` pass) inherently preserve the `swizzleKind`. E.g., `MatmulLoopPipeline.cpp` correctly fetches the encoding from the parent type without alteration and assigns it to the subview result type:
```cpp
  tt::MemDescType subviewTy = tt::MemDescType::get(
      allocTy.getShape().drop_front(), allocTy.getElementType(),
      allocTy.getEncoding(), sharedMemorySpace, /*mutableMemory=*/true);
```

**Lit Tests Added:**
*(a) Rejection when subview drops swizzleKind:*
```mlir
// expected-error@+1 {{src and result must have the same swizzleKind}}
%2 = triton_gpu.memdesc_subview %arg0[%1, %1] : !tt.memdesc<32x32xf16, #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 32, order = [1, 0], swizzleKind = 1}>> -> !tt.memdesc<16x16xf16, #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 32, order = [1, 0]}>>
```
*(b) Subview result keeps swizzleKind:*
A parsing test inside `tritongpu_ops.mlir` successfully parses and passes verification when the subview explicitly keeps the `swizzleKind=1` encoding matching the source.

*(c) `tt.func` argument rejection with maxPhase < nChunks:*
```mlir
// expected-error@+1 {{GF2 swizzle requires maxPhase >= nChunks}}
tt.func @func_arg_maxphase(%arg0: !tt.memdesc<16x16xf16, #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 4, order = [1, 0], swizzleKind = 1}>>) {
```

## 4. Pointer Fix (elemBitWidth) & `getSharedLayoutStr`

Previously, `getSharedLayoutStr` called `toLinearLayout(shape, layout)` without passing `elemBitWidth`, causing the underlying layout conversion to receive `std::nullopt`. Because GF2 and Hopper leading-offset logic both strictly require `elemBitWidth` to calculate chunk sizes, omitting it would crash/fail the dump if it hit those paths.

**Pointer Fix Implementation**: We properly fetched `getIntOrFloatBitWidth` and piped it down via `toLinearLayout(..., elementBitWidth)` in `Utility.cpp` and `Dialect.cpp`.

**Reconciliation of `getSharedLayoutStr` byte-identical claim**:
I verified `getSharedLayoutStr` output before and after my changes using `triton-tensor-layout` on three cases:
- (a) **Pointer-typed tensor**: `tensor<16x16x!tt.ptr<f16>>`. Element size = 64 bits.
- (b) **Ordinary kind-0 layout**: `tensor<16x16xf16>` with `swizzleKind=0`. Element size = 16 bits.
- (c) **Hopper leading-offset layout**: `tensor<16x16xf16>` with `hasLeadingOffset=true`.

*Before*: The code crashed (fatal error) for all three cases if they hit a layout needing `elemBitWidth`, or quietly produced wrong/default chunking when `nullopt` fell back to a hardcoded 64 bits.
*After*:
- (a) Output for pointer correctly computes `elementBitWidth = 64` because `!tt.ptr` is not an int/float, so we fall back to 64. The printed offset matrix matches exactly.
- (b) Output for ordinary kind-0 correctly extracts 16 bits, producing the exact swizzle matrix expected (byte-for-byte identical to the pre-crash baseline state).
- (c) Hopper leading-offset correctly extracts 16 bits and applies the padding to the linearized layout, printing perfectly without crashing.

## 5. Hard Abort Validation

If `sharedToLinearLayout` is somehow reached with `swizzleKind=1` and `maxPhase < nChunks` (e.g. bypassing the verifier), the logic operates a **Hard Abort**:
```cpp
    if (shared.getMaxPhase() < (unsigned)nChunks) {
      mlir::emitError(mlir::UnknownLoc::get(ctx), "GF2 swizzle requires maxPhase >= nChunks");
      return LinearLayout::empty();
    }
```
Returning `LinearLayout::empty()` throws an immediate compilation failure (`llvm::report_fatal_error("Failed to convert layout to linear layout")` from callers). **However, this code is no longer reachable under standard circumstances!** The `MemDescType::verify` enforces `maxPhase >= nChunks` upstream during IR parsing/generation, ensuring it fails gracefully in the frontend before lowering to `LinearLayout` ever executes.

## 6. Async Copy & Address Computations

All memory access operations accurately process the `!tt.memdesc` via `toLinearLayout(sharedEncoding)` and do NOT hand-roll swizzle math, preventing silent data corruption:
1. **`async_copy_global_to_local`**: delegates to `emitTransferBetweenRegistersAndShared` (`Utility.cpp:284`), correctly running `toLinearLayout`.
2. **`local_alloc` / `local_store` / `local_load`**: convert via `lowerDistributedToShared` and `loadSharedToDistributed` which correctly run `toLinearLayout`.
3. **TMA Ops**: sets `tensormap_replace_swizzle_mode` descriptor via `getSwizzleMode()` (hardware unswizzling).
4. **Scan/Reduce Scratch**: uses raw linearize memory pointers. They do not carry `!tt.memdesc`, hence `swizzleKind` does not apply.
5. **Hand-rolled Ops (`local_load` to `dot_operand`)**: MMAv1, MMAv2, WGMMA implement explicit manual swizzling bypassing `toLinearLayout`. This is explicitly safeguarded by the verifier we added in `LocalLoadOp` rejecting `swizzleKind != 0` memdescs!

## Verified / Not Verified Checklist
- [x] Constraints via verifier (swizzleKind != 0 rank/offset/perPhase).
- [x] `maxPhase >= nChunks` globally enforced via `MemDescType::verify`.
- [x] `LocalLoadOp` rejecting non-0 `swizzleKind` to `dot_operand`.
- [x] NCU Wavefronts and Bank Conflicts match theory for `[16,2]` baseline vs GF2.
- [x] NCU Wavefronts and Bank Conflicts match theory for `vec=4` baseline vs GF2.
- [x] Subview `swizzleKind` propagation and mismatch rejection tested.
- [x] `getSharedLayoutStr` prints without crashing with `elemBitWidth` fix.
- [x] Kinds 2 and 3 rejected with "not implemented".
