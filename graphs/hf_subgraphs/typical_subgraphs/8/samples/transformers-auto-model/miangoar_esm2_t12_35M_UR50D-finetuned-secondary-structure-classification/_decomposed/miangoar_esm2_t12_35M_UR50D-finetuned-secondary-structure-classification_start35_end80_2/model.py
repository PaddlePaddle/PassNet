import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = torch.nn.functional.linear(in_4, tmp_2, tmp_1)
        tmp_2 = tmp_1 = None
        tmp_4 = tmp_3.view((128, -1, 20, 24))
        tmp_3 = None
        tmp_5 = tmp_4.transpose(1, 2)
        tmp_4 = None
        tmp_6 = in_6 * 0.2041241452319315
        tmp_7 = torch.arange(64, device=device(type='cuda', index=0))
        tmp_8 = tmp_7.type_as(tmp_0)
        tmp_7 = None
        tmp_9 = torch.outer(tmp_8, tmp_0)
        tmp_8 = tmp_0 = None
        tmp_10 = torch.cat((tmp_9, tmp_9), dim=-1)
        tmp_9 = None
        tmp_11 = tmp_10.to(device(type='cuda', index=0))
        tmp_10 = None
        tmp_12 = tmp_11.cos()
        tmp_13 = tmp_12[None, None, slice(None, None, None), slice(None, None, None)]
        tmp_12 = None
        tmp_14 = tmp_11.sin()
        tmp_11 = None
        tmp_15 = tmp_14[None, None, slice(None, None, None), slice(None, None, None)]
        tmp_14 = None
        tmp_16 = tmp_13[slice(None, None, None), slice(None, None, None), slice(None, 64, None), slice(None, None, None)]
        tmp_17 = tmp_15[slice(None, None, None), slice(None, None, None), slice(None, 64, None), slice(None, None, None)]
        tmp_18 = tmp_6 * tmp_16
        tmp_16 = None
        tmp_19 = tmp_6.chunk(2, dim=-1)
        tmp_6 = None
        tmp_20 = tmp_19[0]
        tmp_21 = tmp_19[1]
        tmp_19 = None
        tmp_22 = -tmp_21
        tmp_21 = None
        tmp_23 = torch.cat((tmp_22, tmp_20), dim=-1)
        tmp_22 = tmp_20 = None
        tmp_24 = tmp_23 * tmp_17
        tmp_23 = tmp_17 = None
        tmp_25 = tmp_18 + tmp_24
        tmp_18 = tmp_24 = None
        tmp_26 = tmp_25.to(dtype=torch.float32)
        tmp_25 = None
        tmp_27 = tmp_13[slice(None, None, None), slice(None, None, None), slice(None, 64, None), slice(None, None, None)]
        tmp_28 = tmp_15[slice(None, None, None), slice(None, None, None), slice(None, 64, None), slice(None, None, None)]
        tmp_29 = in_5 * tmp_27
        tmp_27 = None
        tmp_30 = in_5.chunk(2, dim=-1)
        tmp_31 = tmp_30[0]
        tmp_32 = tmp_30[1]
        tmp_30 = None
        tmp_33 = -tmp_32
        tmp_32 = None
        tmp_34 = torch.cat((tmp_33, tmp_31), dim=-1)
        tmp_33 = tmp_31 = None
        tmp_35 = tmp_34 * tmp_28
        tmp_34 = tmp_28 = None
        tmp_36 = tmp_29 + tmp_35
        tmp_29 = tmp_35 = None
        tmp_37 = tmp_36.to(dtype=torch.float32)
        tmp_36 = None
        tmp_38 = tmp_37.transpose(-1, -2)
        tmp_37 = None
        tmp_39 = torch.matmul(tmp_26, tmp_38)
        tmp_26 = tmp_38 = None
        tmp_40 = tmp_39 + in_3
        tmp_39 = None
        tmp_41 = torch.nn.functional.softmax(tmp_40, dim=-1)
        tmp_40 = None
        tmp_42 = torch.nn.functional.dropout(tmp_41, 0.0, False, False)
        tmp_41 = None
        tmp_43 = tmp_42.to(torch.float32)
        tmp_42 = None
        tmp_44 = torch.matmul(tmp_43, tmp_5)
        tmp_43 = tmp_5 = None
        tmp_45 = tmp_44.permute(0, 2, 1, 3)
        tmp_44 = None
        tmp_46 = tmp_45.contiguous()
        tmp_45 = None
        tmp_47 = tmp_46.view((128, 64, 480))
        tmp_46 = None
        return (tmp_13, tmp_15, tmp_47)