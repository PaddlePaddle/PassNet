import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, in_0, in_1, in_2):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = w_4
        tmp_5 = w_5
        tmp_6 = w_6
        tmp_7 = w_7
        tmp_8 = w_8
        tmp_9 = torch.nn.functional.gelu(in_1)
        tmp_10 = torch.nn.functional.linear(tmp_9, tmp_3, tmp_2)
        tmp_9 = tmp_3 = tmp_2 = None
        tmp_11 = torch.nn.functional.dropout(tmp_10, 0.0, False, False)
        tmp_10 = None
        tmp_12 = tmp_11 + in_0
        tmp_11 = None
        tmp_13 = torch.nn.functional.layer_norm(tmp_12, (32,), tmp_1, tmp_0, 1e-12)
        tmp_12 = tmp_1 = tmp_0 = None
        tmp_14 = tmp_13[slice(None, None, None), 0]
        tmp_15 = torch.nn.functional.linear(tmp_14, tmp_5, tmp_4)
        tmp_14 = tmp_5 = tmp_4 = None
        tmp_16 = torch.tanh(tmp_15)
        tmp_15 = None
        tmp_17 = torch.nn.functional.linear(in_2, tmp_7, None)
        tmp_7 = None
        tmp_18 = torch.nn.functional.linear(tmp_16, tmp_6, None)
        tmp_6 = None
        tmp_19 = tmp_17.norm(p=2, dim=-1, keepdim=True)
        tmp_20 = tmp_17 / tmp_19
        tmp_17 = tmp_19 = None
        tmp_21 = tmp_18.norm(p=2, dim=-1, keepdim=True)
        tmp_22 = tmp_18 / tmp_21
        tmp_18 = tmp_21 = None
        tmp_23 = tmp_8.exp()
        tmp_8 = None
        tmp_24 = tmp_23.to(device=device(type='cuda', index=0))
        tmp_23 = None
        tmp_25 = tmp_20.to(device=device(type='cuda', index=0), dtype=torch.float32)
        tmp_20 = None
        tmp_26 = tmp_25.t()
        tmp_27 = torch.matmul(tmp_22, tmp_26)
        tmp_26 = None
        tmp_28 = tmp_27 * tmp_24
        tmp_27 = tmp_24 = None
        tmp_29 = tmp_28.t()
        return (tmp_13, tmp_16, tmp_22, tmp_25, tmp_28, tmp_29)