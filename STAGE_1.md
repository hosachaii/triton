# Stage 1: GF(2) Swizzle for Shared Memory (Debug Investigation & Verification)

## 1. Clean Rebuild & Library Update
All temporary debug prints (`llvm::errs()`) have been stripped from `LinearLayoutConversions.cpp`. The Python-facing Triton compiler was rebuilt using:
```bash
source ~/.venv/bin/activate && time ninja -j2 -C python/build/cmake.linux-x86_64-cpython-3.12
```
The rebuild successfully updated `libtriton.so` without errors (mtime is newer than the source edit). 

## 2. Explanation of Previous Vectorization Issues
The `test_kernel_vec4_discrim.ttgir` kernel I previously evaluated for `vec=4` had `vec=1, maxPhase=32` specified in its `#triton_gpu.shared` layout attribute (I had failed to adjust it when creating the IR file). 
Since the lowering logic in `emitTransferBetweenRegistersAndShared` uses `shared.getVec()` to determine `maxVecElems`, it forced scalar access `st.shared.u32` and `ld.shared.u32`. Furthermore, the `k=5` shift observed in the PTX (`and.b32 %r25, %r18, 12`) came from the GF(2) implementation deriving `k` dynamically based on the layout's `vec=1` attribute. I have now corrected the IR in `scratch/test_kernel_vec4_discrim.ttgir` to use `#triton_gpu.shared<{vec = 4, perPhase = 1, maxPhase = 8, order = [1, 0]}>`.

## 3. PTX Vectorization Check (`vec=4` kernel)
Compiling `scratch/test_kernel_vec4_discrim.ttgir` with the corrected `vec=4` layout now successfully vectorizes the shared memory instructions. Extracted lines from the resulting PTX (using `TRITON_ALWAYS_COMPILE=1 TRITON_SHARED_SWIZZLE=gf2 python3 scratch/run_ttgir.py scratch/test_kernel_vec4_discrim.ttgir`):
```ptx
	st.shared.v4.u32 	[%r33], {%r1, %r2, %r3, %r4};
	ld.shared.v4.u32 	{%r5, %r6, %r7, %r8}, [%r33];
```

## 4. Verification Status
I ran `ncu` natively (without `sudo`) for the kernel metrics:

- **Verified**: `vec=1` discriminating kernels (`test_kernel_discrim_gf2.ttgir` for `[16,2]` sizePerThread, `test_kernel_default.ttgir` for column layout, `test_kernel_default_row.ttgir` for row layout).
  NCU command pattern:
  ```bash
  TRITON_ALWAYS_COMPILE=1 TRITON_SHARED_SWIZZLE="" ncu --clock-control none --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum -f python scratch/run_ttgir.py <kernel_file>
  ```
  Results for `[16,2]` pair (`scratch/test_kernel_discrim_gf2.ttgir`):
  - **OFF**: 32 ld / 32 st
  - **ON**: 0 ld / 0 st

  Results for Column layout (`scratch/test_kernel_default.ttgir`):
  - **OFF**: 992 ld / 992 st
  - **ON**: 0 ld / 0 st

  Results for Row layout (`scratch/test_kernel_default_row.ttgir`):
  - **OFF**: 0 ld / 0 st
  - **ON**: 0 ld / 0 st

- **Verified**: Vectorization preservation. The modified `scratch/test_kernel_vec4_discrim.ttgir` correctly emits `ld.shared.v4.u32` with GF2 on.

- **Verified**: Physical bank conflicts for `vec>1` match the analytical derivation exactly.
  NCU command:
  ```bash
  TRITON_ALWAYS_COMPILE=1 TRITON_SHARED_SWIZZLE="" ncu --clock-control none --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,sm__cycles_elapsed.avg -k regex:test_kernel_vec4_discrim -f python scratch/run_ttgir.py scratch/test_kernel_vec4_discrim.ttgir
  ```
  Results for `scratch/test_kernel_vec4_discrim.ttgir` (`vec=4, k=3`):
  - **OFF**: 32 ld / 32 st, 6,288 cycles
  - **ON**: 32 ld / 32 st, 7,393 cycles

  As derived, `G(1)=2` leaves row 1 in the same 16-bank half as row 0. Since lanes 0-7 require 128 bytes but touch only 16 banks, it requires 2 wavefronts (a conflict), matching the plain XOR performance for this specific layout. My prediction in the previous run was flawed.
