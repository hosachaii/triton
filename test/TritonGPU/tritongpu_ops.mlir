// RUN: triton-opt %s | triton-opt | FileCheck %s

#shared0 = #triton_gpu.shared<{vec = 8, perPhase = 4, maxPhase = 2, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0], hasLeadingOffset=true}>
// CHECK: #shared{{.*}} = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = false, swizzleKind = 1}>
#shared_gf2 = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 8, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0], swizzleKind = 1}>
// CHECK: #shared{{.*}} = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = false}>
#shared_nogf2 = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 8, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0], swizzleKind = 0}>

module attributes {"triton_gpu.target" = "cuda:0", "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: memdesc
  // CHECK-SAME: !tt.memdesc<1x64x16xf16, #{{.+}}>
  tt.func @memdesc(%d : !tt.memdesc<1x64x16xf16, #shared0>) {
    tt.return
  }

  // CHECK-LABEL: memdesc_gf2
  // CHECK-SAME: !tt.memdesc<1x64x16xf16, #shared1>
  tt.func @memdesc_gf2(%d : !tt.memdesc<1x64x16xf16, #shared_gf2>) {
    tt.return
  }

  // CHECK-LABEL: memdesc_nogf2
  // CHECK-SAME: !tt.memdesc<1x64x16xf16, #shared2>
  tt.func @memdesc_nogf2(%d : !tt.memdesc<1x64x16xf16, #shared_nogf2>) {
    tt.return
  }

  // CHECK-LABEL: subview_gf2
  // CHECK-SAME: !tt.memdesc<16x16xf16, #shared1>
  tt.func @subview_gf2(%a: !tt.memdesc<32x32xf16, #shared_gf2>) {
    // CHECK: triton_gpu.memdesc_subview %{{.*}}[%{{.*}}, %{{.*}}] : !tt.memdesc<32x32xf16, #shared1> -> !tt.memdesc<16x16xf16, #shared1>
    %1 = arith.constant 0 : i32
    %2 = triton_gpu.memdesc_subview %a[%1, %1] : !tt.memdesc<32x32xf16, #shared_gf2> -> !tt.memdesc<16x16xf16, #shared_gf2>
    tt.return
  }
}
