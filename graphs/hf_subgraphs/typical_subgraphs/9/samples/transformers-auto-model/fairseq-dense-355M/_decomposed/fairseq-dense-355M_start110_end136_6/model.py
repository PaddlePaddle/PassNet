import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, in_0, in_1, in_2):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = in_2 * 0.125
        tmp_5 = torch.nn.functional.linear(in_1, tmp_1, tmp_0)
        tmp_1 = tmp_0 = None
        tmp_6 = torch.nn.functional.linear(in_1, tmp_3, tmp_2)
        tmp_3 = tmp_2 = None
        tmp_7 = tmp_5.view(1, 19, -1, 64)
        tmp_5 = None
        tmp_8 = tmp_7.transpose(1, 2)
        tmp_7 = None
        tmp_9 = tmp_6.view(1, 19, -1, 64)
        tmp_6 = None
        tmp_10 = tmp_9.transpose(1, 2)
        tmp_9 = None
        tmp_11 = tmp_4.view(1, 19, 16, 64)
        tmp_4 = None
        tmp_12 = tmp_11.transpose(1, 2)
        tmp_11 = None
        tmp_13 = tmp_12.reshape(16, -1, 64)
        tmp_12 = None
        tmp_14 = tmp_8.reshape(16, -1, 64)
        tmp_8 = None
        tmp_15 = tmp_10.reshape(16, -1, 64)
        tmp_10 = None
        tmp_16 = tmp_14.transpose(1, 2)
        tmp_14 = None
        tmp_17 = torch.bmm(tmp_13, tmp_16)
        tmp_13 = tmp_16 = None
        tmp_18 = tmp_17.view(1, 16, 19, 19)
        tmp_17 = None
        tmp_19 = tmp_18 + in_0
        tmp_18 = None
        tmp_20 = torch.tensor(-65504.0, device=device(type='cuda'))
        tmp_21 = torch.max(tmp_19, tmp_20)
        tmp_19 = tmp_20 = None
        tmp_22 = tmp_21.view(16, 19, 19)
        tmp_21 = None
        tmp_23 = torch.nn.functional.softmax(tmp_22, dim=-1, dtype=torch.float32)
        tmp_22 = None
        tmp_24 = tmp_23.to(torch.float16)
        tmp_23 = None
        tmp_25 = torch.nn.functional.dropout(tmp_24, p=0.1, training=False)
        tmp_24 = None
        tmp_26 = torch.bmm(tmp_25, tmp_15)
        tmp_25 = tmp_15 = None
        tmp_27 = tmp_26.view(1, 16, 19, 64)
        tmp_26 = None
        tmp_28 = tmp_27.transpose(1, 2)
        tmp_27 = None
        tmp_29 = tmp_28.reshape(1, 19, 1024)
        tmp_28 = None
        return (tmp_29,)