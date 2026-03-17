import torch
from torch import device
from torch import inf

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, in_0):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = w_4
        tmp_5 = w_5
        tmp_6 = w_6
        tmp_7 = torch.nn.functional.relu(in_0, inplace=True)
        tmp_8 = torch.conv2d(tmp_7, tmp_4, tmp_3, (1, 1), (0, 0), (1, 1), 1)
        tmp_9 = torch.conv2d(tmp_7, tmp_2, tmp_1, (1, 1), (0, 0), (1, 1), 1)
        tmp_10 = torch.conv2d(tmp_7, tmp_6, tmp_5, (1, 1), (0, 0), (1, 1), 1)
        tmp_11 = torch.functional.einsum('bchw,bciw->bwhi', tmp_8, tmp_9)
        tmp_12 = torch.tensor(-inf)
        tmp_13 = tmp_12.to(device(type='cuda', index=0))
        tmp_12 = None
        tmp_14 = tmp_13.repeat(64)
        tmp_13 = None
        tmp_15 = torch.diag(tmp_14, 0)
        tmp_14 = None
        tmp_16 = tmp_11 + tmp_15
        tmp_11 = tmp_15 = None
        tmp_17 = tmp_16.transpose(1, 2)
        tmp_16 = None
        tmp_18 = torch.functional.einsum('bchw,bchj->bhwj', tmp_8, tmp_9)
        tmp_8 = tmp_9 = None
        tmp_19 = torch.cat([tmp_17, tmp_18], dim=-1)
        tmp_17 = tmp_18 = None
        tmp_20 = torch.nn.functional.softmax(tmp_19, dim=-1)
        tmp_19 = None
        tmp_21 = tmp_20[Ellipsis, slice(None, 64, None)]
        tmp_22 = torch.functional.einsum('bciw,bhwi->bchw', tmp_10, tmp_21)
        tmp_21 = None
        tmp_23 = tmp_20[Ellipsis, slice(64, None, None)]
        tmp_20 = None
        tmp_24 = torch.functional.einsum('bchj,bhwj->bchw', tmp_10, tmp_23)
        tmp_10 = tmp_23 = None
        tmp_22 += tmp_24
        tmp_25 = tmp_22
        tmp_22 = tmp_24 = None
        tmp_26 = tmp_25 * tmp_0
        tmp_25 = None
        tmp_27 = tmp_26 + tmp_7
        tmp_26 = tmp_7 = None
        tmp_28 = tmp_27.contiguous()
        tmp_27 = None
        tmp_29 = torch.conv2d(tmp_28, tmp_4, tmp_3, (1, 1), (0, 0), (1, 1), 1)
        tmp_4 = tmp_3 = None
        tmp_30 = torch.conv2d(tmp_28, tmp_2, tmp_1, (1, 1), (0, 0), (1, 1), 1)
        tmp_2 = tmp_1 = None
        tmp_31 = torch.conv2d(tmp_28, tmp_6, tmp_5, (1, 1), (0, 0), (1, 1), 1)
        tmp_6 = tmp_5 = None
        tmp_32 = torch.functional.einsum('bchw,bciw->bwhi', tmp_29, tmp_30)
        tmp_33 = torch.tensor(-inf)
        tmp_34 = tmp_33.to(device(type='cuda', index=0))
        tmp_33 = None
        tmp_35 = tmp_34.repeat(64)
        tmp_34 = None
        tmp_36 = torch.diag(tmp_35, 0)
        tmp_35 = None
        tmp_37 = tmp_32 + tmp_36
        tmp_32 = tmp_36 = None
        tmp_38 = tmp_37.transpose(1, 2)
        tmp_37 = None
        tmp_39 = torch.functional.einsum('bchw,bchj->bhwj', tmp_29, tmp_30)
        tmp_29 = tmp_30 = None
        tmp_40 = torch.cat([tmp_38, tmp_39], dim=-1)
        tmp_38 = tmp_39 = None
        tmp_41 = torch.nn.functional.softmax(tmp_40, dim=-1)
        tmp_40 = None
        tmp_42 = tmp_41[Ellipsis, slice(None, 64, None)]
        tmp_43 = torch.functional.einsum('bciw,bhwi->bchw', tmp_31, tmp_42)
        tmp_42 = None
        tmp_44 = tmp_41[Ellipsis, slice(64, None, None)]
        tmp_41 = None
        tmp_45 = torch.functional.einsum('bchj,bhwj->bchw', tmp_31, tmp_44)
        tmp_31 = tmp_44 = None
        tmp_43 += tmp_45
        tmp_46 = tmp_43
        tmp_43 = tmp_45 = None
        tmp_47 = tmp_46 * tmp_0
        tmp_46 = tmp_0 = None
        tmp_48 = tmp_47 + tmp_28
        tmp_47 = tmp_28 = None
        tmp_49 = tmp_48.contiguous()
        tmp_48 = None
        return (tmp_49,)