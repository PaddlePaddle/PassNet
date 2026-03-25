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
        tmp_14 = torch.nn.functional.linear(in_0, tmp_7, tmp_6)
        tmp_7 = tmp_6 = None
        tmp_15 = in_1 + tmp_14
        tmp_14 = None
        tmp_16 = torch.nn.functional.layer_norm(tmp_15, (1152,), tmp_1, tmp_0, 1e-06)
        tmp_1 = tmp_0 = None
        tmp_17 = torch.nn.functional.linear(tmp_16, tmp_3, tmp_2)
        tmp_16 = tmp_3 = tmp_2 = None
        tmp_18 = torch.nn.functional.gelu(tmp_17, approximate='tanh')
        tmp_17 = None
        tmp_19 = torch.nn.functional.linear(tmp_18, tmp_5, tmp_4)
        tmp_18 = tmp_5 = tmp_4 = None
        tmp_20 = tmp_15 + tmp_19
        tmp_15 = tmp_19 = None
        tmp_21 = torch.nn.functional.layer_norm(tmp_20, (1152,), tmp_9, tmp_8, 1e-06)
        tmp_20 = tmp_9 = tmp_8 = None
        tmp_22 = tmp_21[slice(None, None, None), -1, slice(None, None, None)]
        tmp_23 = torch.nn.functional.linear(tmp_22, tmp_11, tmp_10)
        tmp_22 = tmp_11 = tmp_10 = None
        tmp_24 = in_2.norm(p=2, dim=-1, keepdim=True)
        tmp_25 = in_2 / tmp_24
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