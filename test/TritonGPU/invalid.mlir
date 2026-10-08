// RUN: triton-opt --split-input-file %s --verify-diagnostics

tt.func public @subview_element_ty(%arg0: !tt.memdesc<8x16xf32>) {
    %zero = arith.constant 0 : i32
    // expected-error @+1 {{element type}}
    %a = triton_gpu.memdesc_subview %arg0[%zero, %zero] : !tt.memdesc<8x16xf32> -> !tt.memdesc<8x16xf16>
    tt.return
}

// -----

tt.func public @too_many_offsets(%arg0: !tt.memdesc<8x16xf32>) {
    %zero = arith.constant 0 : i32
    // expected-error @+1 {{offsets}}
    %a = triton_gpu.memdesc_subview %arg0[%zero, %zero, %zero] : !tt.memdesc<8x16xf32> -> !tt.memdesc<f32>
    tt.return
}

// -----

tt.func public @too_few_offsets(%arg0: !tt.memdesc<8x16xf32>) {
    %zero = arith.constant 0 : i32
    // expected-error @+1 {{offsets}}
    %a = triton_gpu.memdesc_subview %arg0[%zero] : !tt.memdesc<8x16xf32> -> !tt.memdesc<f32>
    tt.return
}

// -----

tt.func public @result_rank_too_large(%arg0: !tt.memdesc<8x16xf32>) {
    %zero = arith.constant 0 : i32
    // expected-error @+1 {{result rank}}
    %a = triton_gpu.memdesc_subview %arg0[%zero, %zero] : !tt.memdesc<8x16xf32> -> !tt.memdesc<3x8x16xf32>
    tt.return
}

// -----

tt.func public @result_dim_too_large(%arg0: !tt.memdesc<8x16xf32>) {
    %zero = arith.constant 0 : i32
    // expected-error @+1 {{result shape}}
    %a = triton_gpu.memdesc_subview %arg0[%zero, %zero] : !tt.memdesc<8x16xf32> -> !tt.memdesc<32xf32>
    tt.return
}

// -----

#mma0 = #triton_gpu.nvidia_mma<{versionMajor=2, warpsPerCTA=[1,1]}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#mma0, kWidth=2}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#mma0, kWidth=2}>
module attributes {"triton_gpu.num-warps" = 1 : i32} {
  tt.func @convert_dot(%A: tensor<16x16xf32, #dot_operand_a>, %B: tensor<16x16xf16, #dot_operand_b>, %C: tensor<16x16xf32, #mma0>) {
    // expected-error@+1 {{element types of operands A and B must have same bit width}}
    %D = tt.dot %A, %B, %C : tensor<16x16xf32, #dot_operand_a> * tensor<16x16xf16, #dot_operand_b> -> tensor<16x16xf32, #mma0>
    tt.return
  }
}

// -----

#mma0 = #triton_gpu.nvidia_mma<{versionMajor=2, warpsPerCTA=[1,1]}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#mma0, kWidth=1}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#mma0, kWidth=2}>
module attributes {"triton_gpu.num-warps" = 1 : i32} {
  tt.func @convert_dot(%A: tensor<16x16xf16>, %B: tensor<16x16xf16, #dot_operand_b>, %C: tensor<16x16xf32, #mma0>) {
    // expected-error@+1 {{mismatching encoding between A and B operands}}
    %D = tt.dot %A, %B, %C : tensor<16x16xf16> * tensor<16x16xf16, #dot_operand_b> -> tensor<16x16xf32, #mma0>
    tt.return
  }
}

// -----

#mma0 = #triton_gpu.nvidia_mma<{versionMajor=2, warpsPerCTA=[1,1]}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#mma0, kWidth=2}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#mma0, kWidth=2}>
module attributes {"triton_gpu.num-warps" = 1 : i32} {
  tt.func @convert_dot(%A: tensor<16x16xf16, #dot_operand_a>, %B: tensor<16x16xf16, #dot_operand_b>, %C: tensor<16x16xf32>) {
    // expected-error@+1 {{miss encoding of C operand}}
    %D = tt.dot %A, %B, %C : tensor<16x16xf16, #dot_operand_a> * tensor<16x16xf16, #dot_operand_b> -> tensor<16x16xf32>
    tt.return
  }
}

// -----

#mma0 = #triton_gpu.nvidia_mma<{versionMajor=2, warpsPerCTA=[1,1]}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#mma0, kWidth=1}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#mma0, kWidth=2}>
module attributes {"triton_gpu.num-warps" = 1 : i32} {
  tt.func @convert_dot(%A: tensor<16x16xf16, #dot_operand_a>, %B: tensor<16x16xf16, #dot_operand_b>, %C: tensor<16x16xf32, #mma0>) {
    // expected-error@+1 {{mismatching kWidth between A and B operands}}
    %D = tt.dot %A, %B, %C : tensor<16x16xf16, #dot_operand_a> * tensor<16x16xf16, #dot_operand_b> -> tensor<16x16xf32, #mma0>
    tt.return
  }
}

// -----

// expected-error@+2 {{memdesc with swizzleKind != 0 must not be consumed by a dot operand load}}
tt.func public @invalid_dot_operand_load(%arg0: !tt.memdesc<16x16xf16, #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], swizzleKind = 1}>>) {
  %1 = triton_gpu.local_load %arg0 : !tt.memdesc<16x16xf16, #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], swizzleKind = 1}>> -> tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [1, 0]}>}>>
  tt.return
}

// -----

#shared = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0], swizzleKind = 1}>
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [1, 0]}>

module attributes {"triton_gpu.num-warps" = 1 : i32} {
  tt.func @kernel(%arg0: tensor<16x16xf16, #blocked>) {
    // expected-error@+1 {{GF2 swizzle requires maxPhase >= nChunks}}
    %6 = triton_gpu.local_alloc %arg0 : (tensor<16x16xf16, #blocked>) -> !tt.memdesc<16x16xf16, #shared>
    tt.return
  }
}

// -----

// expected-error@+1 {{swizzleKind != 0 requires rank 2}}
#shared_rank3 = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 8, order = [2, 1, 0], swizzleKind = 1}>

// -----

// expected-error@+1 {{swizzleKind != 0 requires perPhase = 1}}
#shared_perphase2 = #triton_gpu.shared<{vec = 1, perPhase = 2, maxPhase = 8, order = [1, 0], swizzleKind = 1}>

// -----

// expected-error@+1 {{swizzleKind != 0 requires hasLeadingOffset = false}}
#shared_offset = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = true, swizzleKind = 1}>

// -----

// expected-error@+2 {{GF2 swizzle requires maxPhase >= nChunks}}
tt.func @local_alloc_maxphase(%arg0: tensor<16x16xf16, #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [1, 0]}>>) {
  %1 = triton_gpu.local_alloc %arg0 : (tensor<16x16xf16, #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [1, 0]}>>) -> !tt.memdesc<16x16xf16, #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 4, order = [1, 0], swizzleKind = 1}>>
  tt.return
}

// -----

// expected-error@+1 {{GF2 swizzle requires maxPhase >= nChunks}}
tt.func @func_arg_maxphase(%arg0: !tt.memdesc<16x16xf16, #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 4, order = [1, 0], swizzleKind = 1}>>) {
  tt.return
}

// -----

tt.func @subview_swizzlekind_mismatch(%arg0: !tt.memdesc<32x32xf16, #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 64, order = [1, 0], swizzleKind = 1}>>) {
  %1 = arith.constant 0 : i32
  // expected-error@+1 {{src and result must have the same swizzleKind}}
  %2 = triton_gpu.memdesc_subview %arg0[%1, %1] : !tt.memdesc<32x32xf16, #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 64, order = [1, 0], swizzleKind = 1}>> -> !tt.memdesc<16x16xf16, #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 64, order = [1, 0], swizzleKind = 0}>>
  tt.return
}
// -----

// expected-error@+2 {{memdesc with swizzleKind != 0 must not be consumed by a dot operand load}}
tt.func @local_load_dot(%arg1: !tt.memdesc<16x16xf16, #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], swizzleKind = 1}>>) {
  %2 = triton_gpu.local_load %arg1 : !tt.memdesc<16x16xf16, #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], swizzleKind = 1}>> -> tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #triton_gpu.nvidia_mma<{versionMajor = 2, warpsPerCTA = [1, 1]}>, kWidth=2}>>
  tt.return
}


// -----

tt.func @memdesc_subview_swizzle_drop(%arg0: !tt.memdesc<32x32xf16, #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], swizzleKind = 1}>>) {
  %1 = arith.constant 0 : i32
  // expected-error@+1 {{src and result must have the same swizzleKind}}
  %2 = triton_gpu.memdesc_subview %arg0[%1, %1] : !tt.memdesc<32x32xf16, #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], swizzleKind = 1}>> -> !tt.memdesc<16x16xf16, #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 8, order = [1, 0]}>>
  tt.return
}
// -----

// expected-error@+1 {{swizzleKind 2 and 3 are not implemented}}
tt.func @memdesc_swizzlekind_2(%arg0: !tt.memdesc<32x32xf16, #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 32, order = [1, 0], swizzleKind = 2}>>) {
  tt.return
}

// -----

// expected-error@+1 {{swizzleKind 2 and 3 are not implemented}}
tt.func @memdesc_swizzlekind_3(%arg0: !tt.memdesc<32x32xf16, #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 32, order = [1, 0], swizzleKind = 3}>>) {
  tt.return
}
