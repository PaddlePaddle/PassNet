import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, w_0, w_1, w_2):
        tmp_0 = torch.nn.functional.embedding(in_1, w_0, None, None, 2.0, False, False)
        tmp_1 = tmp_0.to(device(type='cuda', index=0))
        tmp_0 = None
        tmp_2 = in_0 + tmp_1
        tmp_1 = None
        tmp_3 = torch.nn.functional.layer_norm(tmp_2, (16,), w_2, w_1, 1e-05)
        return (tmp_2, tmp_3)