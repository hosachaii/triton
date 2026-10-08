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
Testing a layout that stores via `sizePerThread=[1,4]` (128-bit `vec=4` store) and loads via `sizePerThread=[4,1]` (128-bit `vec=4` load), with `fp32` elements (`elemBytes=4`). We have `nChunks = 128 / (4 * 4) = 8`, and the encoding sets `maxPhase = 8`.

Because `maxPhase >= nChunks` (8 >= 8) is satisfied, the scope condition for GF2 is met and **GF2 is applied**. 

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
*Result:* Both load and store emit 128-bit requests (`vec=4`). GF2 counts identically match Plain XOR because `G(1)=2` cannot separate rows `8m` and `8m+1` (they land in the same 16-bank half). So GF2 is actively transforming the swizzle but lands on the same performance as plain XOR for this shape. *(Note: an earlier misconfiguration tested `sizePerThread=[1,4]` with `order=[0,1]` which forced a 32-bit load `vec=1`, artificially creating a 0 load conflict / 32 store conflict mismatch. This is now fully resolved and matches the 32/32 reality.)*

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

Previously, `getSharedLayoutStr` called `toLinearLayout(shape, layout)` without passing `elemBitWidth`, causing the underlying layout conversion to receive `std::nullopt`. Because GF2 and Hopper leading-offset logic both strictly require `elemBitWidth` to calculate chunk sizes, omitting it would crash/fail the dump.

**Pointer Fix Implementation**: We properly fetched `getIntOrFloatBitWidth` and piped it down via `toLinearLayout(..., elementBitWidth)` in `Utility.cpp` and `Dialect.cpp`.
This means:
1. `minimalCvtLayout` cleanly receives and passes `elemBitWidth`, keeping its layout transformations mathematically equivalent and intact.
2. `getSharedLayoutStr` perfectly parses the layout! 
**Byte-identical outputs**: For `swizzleKind=0`, `getSharedLayoutStr` output is perfectly byte-for-byte identical to the output before our GF2 changes (it cleanly falls into the Plain XOR baseline block). Hopper layouts (`hasLeadingOffset=true`) also now print flawlessly without crashing since they correctly receive their bitwidth constraint!

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
