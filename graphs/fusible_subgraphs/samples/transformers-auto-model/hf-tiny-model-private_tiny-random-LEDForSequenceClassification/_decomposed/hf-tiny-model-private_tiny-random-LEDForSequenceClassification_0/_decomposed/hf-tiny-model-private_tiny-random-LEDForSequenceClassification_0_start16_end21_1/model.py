import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self, in_0, w_0, w_1, w_2):
        tmp_0 = torch.arange(0, 22, dtype=torch.int64, device=device(type='cuda', index=0))
        tmp_1 = torch.nn.functional.embedding(tmp_0, w_0, None, None, 2.0, False, False)
        tmp_0 = None
        tmp_2 = in_0 + tmp_1
        tmp_1 = None
        tmp_3 = torch.nn.functional.layer_norm(tmp_2, (16,), w_2, w_1, 1e-05)
        tmp_2 = None
        tmp_4 = torch.nn.functional.dropout(tmp_3, p=0.1, training=False)
        tmp_3 = None
        return (tmp_4,)