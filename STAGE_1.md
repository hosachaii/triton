# Stage 1: GF(2) Swizzle for Shared Memory (Debug Investigation & Verification)

## 1. Clean Rebuild & Library Update
All temporary debug prints (`llvm::errs()`) have been stripped from `LinearLayoutConversions.cpp`. The Python-facing Triton compiler was rebuilt using:
```bash
source ~/.venv/bin/activate && time ninja -j2 -C python/build/cmake.linux-x86_64-cpython-3.12
```
The rebuild successfully updated `libtriton.so` without errors.

## 2. Scope Condition and `#shared` Attributes
The exact condition guarding the `gf2` path in `LinearLayoutConversions.cpp` is:
```cpp
const char *swizzleEnv = std::getenv("TRITON_SHARED_SWIZZLE");
bool useGF2 = swizzleEnv && std::string(swizzleEnv) == "gf2";
// ...
if (useGF2 && rank == 2 && shared.getPerPhase() == 1 && elemBitWidth.has_value()) {
    int elemBytes = *elemBitWidth / 8;
    int vec = shared.getVec();
    int nChunks = 128 / (vec * elemBytes);
    if (nChunks > 0 && llvm::isPowerOf2_32(nChunks) && shared.getMaxPhase() >= nChunks) {
// ...
```
This logic already enforces `maxPhase >= nChunks`. If the condition fails, it safely falls through to the existing Plain XOR path.

The `#shared` attributes from the `.ttgir` files tested:
- `scratch/test_kernel_default.ttgir` (Column Read): `#triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>`
- `scratch/test_kernel_default_row.ttgir` (Row Read): `#triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>`
- `scratch/test_kernel_discrim_gf2.ttgir` (`[16,2]`): `#triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 32, order = [1, 0]}>`
- `scratch/test_kernel_vec4_discrim.ttgir` (`vec=4`): `#triton_gpu.shared<{vec = 4, perPhase = 1, maxPhase = 8, order = [1, 0]}>`

*Note on Column Read Conflicts*: The `test_kernel_default.ttgir` file specifies `maxPhase=1`. Because `1 < 32` (the `nChunks` for `vec=1`), GF2 correctly rejects it and falls through to Plain XOR. Plain XOR with `maxPhase=1` results in no swizzling (`P_col = L_col ^ 0 = L_col`), generating 992 conflicts regardless of the switch state. 
The 0 conflict result reported previously for the `ON` run came from profiling `test_kernel_gf2.ttgir` (which forces `maxPhase=32`), where *both* Plain XOR and GF2 resolve to 0 conflicts for a pure column read.

## 3. Environment Variable Handling
An empty string `TRITON_SHARED_SWIZZLE=""` defines the variable, but `std::string(swizzleEnv) == "gf2"` evaluates to `false`. Being entirely unset makes `std::getenv` return `nullptr`, which also short-circuits `useGF2` to `false`. Both scenarios correctly bypass GF2 and produce the exact same outcome. The OFF runs below strictly use `unset TRITON_SHARED_SWIZZLE`.

## 4. Final Lit and GTest Baseline
Running the test suites on the clean final build:
```bash
unset TRITON_SHARED_SWIZZLE
BUILD_DIR=python/build/cmake.linux-x86_64-cpython-3.12
lit -v --timeout 120 $BUILD_DIR/test/
./$BUILD_DIR/unittest/Dialect/TritonGPU/LinearLayoutConversions
```
- **Lit Tests**: `Total Discovered Tests: 79, Passed: 79 (100.00%)`
- **GF2Swizzle GTest**: `[  PASSED  ] 61 tests.` (including `GF2Swizzle`)

## 5. PTX Vectorization Check (`vec=4` kernel)
Compiling `scratch/test_kernel_vec4_discrim.ttgir` successfully vectorizes the shared memory instructions. 
Extracted PTX lines:

**OFF** (`unset TRITON_SHARED_SWIZZLE; python scratch/run_ttgir.py scratch/test_kernel_vec4_discrim.ttgir`):
```ptx
	xor.b32  	%r25, %r24, %r18;
	or.b32  	%r26, %r25, %r20;
	shl.b32 	%r27, %r26, 2;
	mov.u32 	%r28, global_smem;
	add.s32 	%r29, %r28, %r27;
	st.shared.v4.u32 	[%r29], {%r1, %r2, %r3, %r4};
```

**ON** (`TRITON_SHARED_SWIZZLE=gf2 python scratch/run_ttgir.py scratch/test_kernel_vec4_discrim.ttgir`):
```ptx
	xor.b32  	%r29, %r28, %r26;
	or.b32  	%r30, %r29, %r20;
	shl.b32 	%r31, %r30, 2;
	mov.u32 	%r32, global_smem;
	add.s32 	%r33, %r32, %r31;
	st.shared.v4.u32 	[%r33], {%r1, %r2, %r3, %r4};
```

## 6. Verification Status
I ran `ncu` natively (without `sudo`) for the kernel metrics using `unset TRITON_SHARED_SWIZZLE` for OFF runs.

- **Verified**: Vectorization preservation. The modified `scratch/test_kernel_vec4_discrim.ttgir` correctly emits `st.shared.v4.u32` with GF2 on.
- **Verified**: Physical bank conflicts for `vec>1` match the analytical derivation exactly.
  Results for `scratch/test_kernel_vec4_discrim.ttgir` (`vec=4, k=3`):
  - **OFF**: 32 ld / 32 st
  - **ON**: 32 ld / 32 st
  As derived, `G(1)=2` leaves row 1 in the same 16-bank half as row 0. Since lanes 0-7 require 128 bytes but touch only 16 banks, it requires 2 wavefronts (a conflict), matching the plain XOR performance for this specific layout.
