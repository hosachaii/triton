
import torch, triton
k = triton.compile('scratch/stage8/test_p3_vec4_gf2.ttgir')
x = torch.randn((32, 32), dtype=torch.float32, device='cuda')
y = torch.empty_like(x)
for _ in range(5):
    k[(1, 1, 1)](x, y)
torch.cuda.synchronize()
