
import torch
import triton

M, N, K = 64, 64, 64
a = torch.randn((M, K), dtype=torch.float16, device='cuda')
b = torch.randn((K, N), dtype=torch.float16, device='cuda')
c = torch.empty((M, N), dtype=torch.float16, device='cuda')

# Stock P1 (with convert_layout in epilogue)
k1 = triton.compile('scratch/stage7/p1_stock.ttgir')
k1[(2, 2, 1)](a, b, c, M, N, K, K, N, N)

# Control P1 (without convert_layout in epilogue)
k2 = triton.compile('scratch/stage8/ctrl_no_convert.ttgir')
k2[(2, 2, 1)](a, b, c, M, N, K, K, N, N)
