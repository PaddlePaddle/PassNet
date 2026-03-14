import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, in_0, in_1, in_2):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = w_4
        tmp_5 = w_5
        tmp_6 = w_6
        tmp_7 = w_7
        tmp_8 = w_8
        tmp_9 = w_9
        tmp_10 = w_10
        tmp_11 = w_11
        tmp_12 = w_12
        tmp_13 = w_13
        tmp_14 = in_0
        tmp_15 = torch.nn.functional.linear(in_1, tmp_7, tmp_6)
        tmp_7 = tmp_6 = None
        tmp_16 = in_2 + tmp_15
        tmp_15 = None
        tmp_17 = torch.nn.functional.layer_norm(tmp_16, (768,), tmp_1, tmp_0, 1e-06)
        tmp_1 = tmp_0 = None
        tmp_18 = torch.nn.functional.linear(tmp_17, tmp_3, tmp_2)
        tmp_17 = tmp_3 = tmp_2 = None
        tmp_19 = torch.nn.functional.gelu(tmp_18, approximate='tanh')
        tmp_18 = None
        tmp_20 = torch.nn.functional.linear(tmp_19, tmp_5, tmp_4)
        tmp_19 = tmp_5 = tmp_4 = None
        tmp_21 = tmp_16 + tmp_20
        tmp_16 = tmp_20 = None
        tmp_22 = torch.nn.functional.layer_norm(tmp_21, (768,), tmp_9, tmp_8, 1e-06)
        tmp_21 = tmp_9 = tmp_8 = None
        tmp_23 = tmp_22[slice(None, None, None), -1, slice(None, None, None)]
        tmp_24 = torch.nn.functional.linear(tmp_23, tmp_11, tmp_10)
        tmp_23 = tmp_11 = tmp_10 = None
        tmp_25 = tmp_14.norm(p=2, dim=-1, keepdim=True)
        tmp_26 = tmp_14 / tmp_25
        tmp_14 = tmp_25 = None
        tmp_27 = tmp_24.norm(p=2, dim=-1, keepdim=True)
        tmp_28 = tmp_24 / tmp_27
        tmp_27 = None
        tmp_29 = tmp_26.t()
        tmp_30 = tmp_29.to(device(type='cuda'))
        tmp_29 = None
        tmp_31 = torch.matmul(tmp_28, tmp_30)
        tmp_30 = None
        tmp_32 = tmp_13.to(device(type='cuda'))
        tmp_13 = None
        tmp_33 = tmp_12.to(device(type='cuda'))
        tmp_12 = None
        tmp_34 = tmp_32.exp()
        tmp_32 = None
        tmp_35 = tmp_31 * tmp_34
        tmp_31 = tmp_34 = None
        tmp_36 = tmp_35 + tmp_33
        tmp_35 = tmp_33 = None
        tmp_37 = tmp_36.t()
        return (tmp_22, tmp_24, tmp_26, tmp_28, tmp_36, tmp_37)