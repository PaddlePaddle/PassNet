import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = in_4
        tmp_5 = in_5
        tmp_6 = in_6
        tmp_7 = in_7
        tmp_8 = in_8
        tmp_9 = in_9
        tmp_10 = in_10
        tmp_11 = in_11
        tmp_12 = in_12
        tmp_13 = in_13
        tmp_14 = torch.nn.functional.linear(in_14, tmp_7, tmp_6)
        tmp_7 = tmp_6 = None
        tmp_15 = in_15 + tmp_14
        tmp_14 = None
        tmp_16 = torch.nn.functional.layer_norm(tmp_15, (768,), tmp_1, tmp_0, 1e-06)
        tmp_1 = tmp_0 = None
        tmp_17 = torch.nn.functional.linear(tmp_16, tmp_3, tmp_2)
        tmp_16 = tmp_3 = tmp_2 = None
        tmp_18 = torch.nn.functional.gelu(tmp_17, approximate='tanh')
        tmp_17 = None
        tmp_19 = torch.nn.functional.linear(tmp_18, tmp_5, tmp_4)
        tmp_18 = tmp_5 = tmp_4 = None
        tmp_20 = tmp_15 + tmp_19
        tmp_15 = tmp_19 = None
        tmp_21 = torch.nn.functional.layer_norm(tmp_20, (768,), tmp_9, tmp_8, 1e-06)
        tmp_20 = tmp_9 = tmp_8 = None
        tmp_22 = tmp_21[slice(None, None, None), -1, slice(None, None, None)]
        tmp_23 = torch.nn.functional.linear(tmp_22, tmp_11, tmp_10)
        tmp_22 = tmp_11 = tmp_10 = None
        tmp_24 = in_16.norm(p=2, dim=-1, keepdim=True)
        tmp_25 = in_16 / tmp_24
        tmp_24 = None
        tmp_26 = tmp_23.norm(p=2, dim=-1, keepdim=True)
        tmp_27 = tmp_23 / tmp_26
        tmp_26 = None
        tmp_28 = tmp_25.t()
        tmp_29 = tmp_28.to(device(type='cuda'))
        tmp_28 = None
        tmp_30 = torch.matmul(tmp_27, tmp_29)
        tmp_29 = None
        tmp_31 = tmp_13.to(device(type='cuda'))
        tmp_13 = None
        tmp_32 = tmp_12.to(device(type='cuda'))
        tmp_12 = None
        tmp_33 = tmp_31.exp()
        tmp_31 = None
        tmp_34 = tmp_30 * tmp_33
        tmp_30 = tmp_33 = None
        tmp_35 = tmp_34 + tmp_32
        tmp_34 = tmp_32 = None
        tmp_36 = tmp_35.t()
        return (tmp_21, tmp_23, tmp_25, tmp_27, tmp_35, tmp_36)