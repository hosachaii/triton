# Stage 1: GF(2) Swizzle for Shared Memory (Debug Investigation & Verification)

## 1. Clean Rebuild & Library Update
All temporary debug prints (`llvm::errs()`) have been stripped from `LinearLayoutConversions.cpp`. The Python-facing Triton compiler was rebuilt using:
```bash
source ~/.venv/bin/activate && time ninja -j2 -C python/build/cmake.linux-x86_64-cpython-3.12
```
The rebuild successfully updated `libtriton.so` without errors. Git diff confirms that only the core GF2 logic and the unit test remain.

## 2. Testing Results
With the clean build, I re-ran the testing suite to confirm no regressions and baseline integrity:
- **GTest**: `GF2Swizzle` unit test passed (1/1). The test brute-force verifies bijectivity for these combinations:
  - `fp32 vec=1` (`k=5`, maxPhase=32, vec=1)
  - `fp32 vec=4` (`k=3`, maxPhase=8, vec=4)
  - `fp16 vec=2` (`k=5`, maxPhase=32, vec=2)
  - `fp16 vec=8` (`k=3`, maxPhase=8, vec=8)
- **Lit Tests (Baseline OFF)**: The drop from 79 tests to 56 was because I only passed `test/TritonGPU` and `test/Conversion` directories to `lit` instead of the whole `test/` directory. Running the exact baseline command `lit -v --timeout 120 python/build/cmake.linux-x86_64-cpython-3.12/test/` correctly yields: `Total Discovered Tests: 79, Passed: 79 (100.00%)`.

## 3. PTX Vectorization Check (`vec=4` kernel)
Compiling a kernel with `vec=4` (`#triton_gpu.shared<{vec = 4, ...}>`) and `TRITON_SHARED_SWIZZLE=gf2` successfully preserves vectorized shared memory instructions. Extracted lines from the resulting PTX:
```ptx
	st.shared.v4.u32 	[%r34], {%r1, %r2, %r3, %r4};
	st.shared.v4.u32 	[%r41], {%r5, %r6, %r7, %r8};
	ld.shared.v4.u32 	{%r13, %r14, %r15, %r16}, [%r41];
	ld.shared.v4.u32 	{%r9, %r10, %r11, %r12}, [%r34];
```

## 4. NCU Commands for `vec>1` Discriminating Kernel
I generated a discriminator kernel `scratch/test_kernel_vec4_discrim.ttgir` with `vec=4`, `sizePerThread=[1,4]`, `threadsPerWarp=[8,4]`, and `warpsPerCTA=[4,2]` for a `32x32` tile. 
Its `#shared` attribute for the OFF run is `#triton_gpu.shared<{vec = 4, perPhase = 1, maxPhase = 8, order = [1, 0]}>`. 

**Predicted Counts & Derivation (lanes 0-7, one 128-byte access group):**
- Lane 0 touches row 0, chunk 0 -> banks 0-3
- Lane 1 touches row 0, chunk 1 -> banks 4-7
- Lane 2 touches row 0, chunk 2 -> banks 8-11
- Lane 3 touches row 0, chunk 3 -> banks 12-15
- Lane 4 touches row 1, chunk 0 -> Plain XOR: banks 4-7. GF2 (G(1)=2 -> col_basis=8): banks 8-11
- Lane 5 touches row 1, chunk 1 -> Plain XOR: banks 0-3. GF2: banks 12-15
- Lane 6 touches row 1, chunk 2 -> Plain XOR: banks 12-15. GF2: banks 0-3
- Lane 7 touches row 1, chunk 3 -> Plain XOR: banks 8-11. GF2: banks 4-7

**Wavefront count for 128 bytes (minimum 1 wavefront):**
- **Plain XOR**: Lanes 0-3 and 4-7 both perfectly cover banks 0-15. Banks 0-15 provide 64 bytes/transaction. 128 bytes requires **2 wavefronts**.
- **GF2**: Lanes 0-3 and 4-7 ALSO both perfectly cover banks 0-15 (because an offset of 8 stays within 0-15). 128 bytes requires **2 wavefronts**.

My prior prediction was incorrect. Because $G(1)=2$ only shifts the chunk offset to 2 (8 banks), rows 0 and 1 still share the same 16-bank half. Neither layout achieves the hardware minimum for this specific `vec=4` setup!

## 5. Verification Status
The previous failure to see the PTX diff in earlier runs was due to a stale `libtriton.so` compilation (likely my interrupted build), as the discriminating runs showed the env var was successfully reaching the process under `sudo`.
- **Verified**: Core fp32 `vec=1` case conflicts (0/0) and functionality.
- **Verified**: Vectorization preservation (`ld.shared.v4` generated correctly).
- **Not Verified**: Physical bank conflicts for `vec>1` on the compiled path (waiting on user execution to confirm if it matches the derivation).
