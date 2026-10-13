#!/usr/bin/env python3
import os
import torch
import triton

# Compile the 3 kernels
k_stock = triton.compile("scratch/stage8/ctrl_stock.ttgir")
k_p1_m4 = triton.compile("scratch/stage8/ctrl_p1_m4.ttgir")
k_p1_m2 = triton.compile("scratch/stage8/ctrl_p1_m2.ttgir")

# Allocate tensors for 32x32 matmul
# Arguments: a, b, c, M, N, K, stride_am, stride_bk, stride_cm
M, N, K = 32, 32, 32
a = torch.randn((M, K), dtype=torch.float16, device="cuda")
b = torch.randn((K, N), dtype=torch.float16, device="cuda")
c = torch.empty((M, N), dtype=torch.float16, device="cuda")

stride_am, stride_ak = a.stride(0), a.stride(1)
stride_bk, stride_bn = b.stride(0), b.stride(1)
stride_cm, stride_cn = c.stride(0), c.stride(1)

# Warmup
for _ in range(5):
    k_stock[(1, 1, 1)](a, b, c, M, N, K, stride_am, stride_bk, stride_cm)
    k_p1_m4[(1, 1, 1)](a, b, c, M, N, K, stride_am, stride_bk, stride_cm)
    k_p1_m2[(1, 1, 1)](a, b, c, M, N, K, stride_am, stride_bk, stride_cm)
torch.cuda.synchronize()

# Single launch each for NCU profiling
k_stock[(1, 1, 1)](a, b, c, M, N, K, stride_am, stride_bk, stride_cm)
torch.cuda.synchronize()

k_p1_m4[(1, 1, 1)](a, b, c, M, N, K, stride_am, stride_bk, stride_cm)
torch.cuda.synchronize()

k_p1_m2[(1, 1, 1)](a, b, c, M, N, K, stride_am, stride_bk, stride_cm)
torch.cuda.synchronize()
print("Run completed successfully!")
