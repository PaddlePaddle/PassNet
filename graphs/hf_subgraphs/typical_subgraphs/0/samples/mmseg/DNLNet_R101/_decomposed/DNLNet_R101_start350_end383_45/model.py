import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = in_4
        tmp_5 = in_5
        tmp_6 = in_6
        tmp_7 = in_7
        tmp_8 = torch.nn.functional.relu(in_8, inplace=True)
        tmp_9 = torch.conv2d(tmp_8, tmp_3, tmp_2, (1, 1), (0, 0), (1, 1), 1)
        tmp_3 = tmp_2 = None
        tmp_10 = tmp_9.view(1, 256, -1)
        tmp_9 = None
        tmp_11 = tmp_10.permute(0, 2, 1)
        tmp_10 = None
        tmp_12 = torch.conv2d(tmp_8, tmp_7, tmp_6, (1, 1), (0, 0), (1, 1), 1)
        tmp_7 = tmp_6 = None
        tmp_13 = tmp_12.view(1, 256, -1)
        tmp_12 = None
        tmp_14 = tmp_13.permute(0, 2, 1)
        tmp_13 = None
        tmp_15 = torch.conv2d(tmp_8, tmp_5, tmp_4, (1, 1), (0, 0), (1, 1), 1)
        tmp_5 = tmp_4 = None
        tmp_16 = tmp_15.view(1, 256, -1)
        tmp_15 = None
        tmp_17 = tmp_14.mean(dim=-2, keepdim=True)
        tmp_14 -= tmp_17
        tmp_18 = tmp_14
        tmp_14 = tmp_17 = None
        tmp_19 = tmp_16.mean(dim=-1, keepdim=True)
        tmp_16 -= tmp_19
        tmp_20 = tmp_16
        tmp_16 = tmp_19 = None
        tmp_21 = torch.matmul(tmp_18, tmp_20)
        tmp_18 = tmp_20 = None
        tmp_22 = torch.tensor(256, dtype=torch.float32, device=device(type='cuda', index=0))
        tmp_23 = torch.tensor(0.5, device=device(type='cuda', index=0))
        tmp_24 = tmp_22 ** tmp_23
        tmp_22 = tmp_23 = None
        tmp_21 /= tmp_24
        tmp_25 = tmp_21
        tmp_21 = tmp_24 = None
        tmp_26 = torch.tensor(0.05, device=device(type='cuda', index=0))
        tmp_25 /= tmp_26
        tmp_27 = tmp_25
        tmp_25 = tmp_26 = None
        tmp_28 = tmp_27.softmax(dim=-1)
        tmp_27 = None
        tmp_29 = torch.matmul(tmp_28, tmp_11)
        tmp_28 = None
        tmp_30 = tmp_29.permute(0, 2, 1)
        tmp_29 = None
        tmp_31 = tmp_30.contiguous()
        tmp_30 = None
        tmp_32 = tmp_31.reshape(1, 256, 64, 64)
        tmp_31 = None
        tmp_33 = torch.conv2d(tmp_8, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
        tmp_1 = tmp_0 = None
        tmp_34 = tmp_33.view(1, 1, -1)
        tmp_33 = None
        tmp_35 = tmp_34.softmax(dim=-1)
        tmp_34 = None
        tmp_36 = torch.matmul(tmp_35, tmp_11)
        tmp_35 = tmp_11 = None
        tmp_37 = tmp_36.permute(0, 2, 1)
        tmp_36 = None
        tmp_38 = tmp_37.contiguous()
        tmp_37 = None
        tmp_39 = tmp_38.reshape(1, 256, 1, 1)
        tmp_38 = None
        tmp_40 = tmp_32 + tmp_39
        tmp_32 = tmp_39 = None
        return (tmp_40, tmp_8)