import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4):
        tmp_0 = in_0
        tmp_1 = in_3 * 0.125
        tmp_2 = torch.arange(128, device=device(type='cuda', index=0))
        tmp_3 = tmp_2.type_as(tmp_0)
        tmp_2 = None
        tmp_4 = torch.outer(tmp_3, tmp_0)
        tmp_3 = tmp_0 = None
        tmp_5 = torch.cat((tmp_4, tmp_4), dim=-1)
        tmp_4 = None
        tmp_6 = tmp_5.to(device(type='cuda', index=0))
        tmp_5 = None
        tmp_7 = tmp_6.cos()
        tmp_8 = tmp_7[None, None, slice(None, None, None), slice(None, None, None)]
        tmp_7 = None
        tmp_9 = tmp_6.sin()
        tmp_6 = None
        tmp_10 = tmp_9[None, None, slice(None, None, None), slice(None, None, None)]
        tmp_9 = None
        tmp_11 = tmp_8[slice(None, None, None), slice(None, None, None), slice(None, 128, None), slice(None, None, None)]
        tmp_12 = tmp_10[slice(None, None, None), slice(None, None, None), slice(None, 128, None), slice(None, None, None)]
        tmp_13 = tmp_1 * tmp_11
        tmp_11 = None
        tmp_14 = tmp_1.chunk(2, dim=-1)
        tmp_1 = None
        tmp_15 = tmp_14[0]
        tmp_16 = tmp_14[1]
        tmp_14 = None
        tmp_17 = -tmp_16
        tmp_16 = None
        tmp_18 = torch.cat((tmp_17, tmp_15), dim=-1)
        tmp_17 = tmp_15 = None
        tmp_19 = tmp_18 * tmp_12
        tmp_18 = tmp_12 = None
        tmp_20 = tmp_13 + tmp_19
        tmp_13 = tmp_19 = None
        tmp_21 = tmp_20.to(dtype=torch.float32)
        tmp_20 = None
        tmp_22 = tmp_8[slice(None, None, None), slice(None, None, None), slice(None, 128, None), slice(None, None, None)]
        tmp_23 = tmp_10[slice(None, None, None), slice(None, None, None), slice(None, 128, None), slice(None, None, None)]
        tmp_24 = in_2 * tmp_22
        tmp_22 = None
        tmp_25 = in_2.chunk(2, dim=-1)
        tmp_26 = tmp_25[0]
        tmp_27 = tmp_25[1]
        tmp_25 = None
        tmp_28 = -tmp_27
        tmp_27 = None
        tmp_29 = torch.cat((tmp_28, tmp_26), dim=-1)
        tmp_28 = tmp_26 = None
        tmp_30 = tmp_29 * tmp_23
        tmp_29 = tmp_23 = None
        tmp_31 = tmp_24 + tmp_30
        tmp_24 = tmp_30 = None
        tmp_32 = tmp_31.to(dtype=torch.float32)
        tmp_31 = None
        tmp_33 = tmp_32.transpose(-1, -2)
        tmp_32 = None
        tmp_34 = torch.matmul(tmp_21, tmp_33)
        tmp_21 = tmp_33 = None
        tmp_35 = tmp_34 + in_1
        tmp_34 = None
        tmp_36 = torch.nn.functional.softmax(tmp_35, dim=-1)
        tmp_35 = None
        tmp_37 = torch.nn.functional.dropout(tmp_36, 0.0, False, False)
        tmp_36 = None
        tmp_38 = tmp_37.to(torch.float32)
        tmp_37 = None
        tmp_39 = torch.matmul(tmp_38, in_4)
        tmp_38 = None
        tmp_40 = tmp_39.permute(0, 2, 1, 3)
        tmp_39 = None
        tmp_41 = tmp_40.contiguous()
        tmp_40 = None
        tmp_42 = tmp_41.view((64, 128, 1280))
        tmp_41 = None
        return (tmp_8, tmp_10, tmp_42)