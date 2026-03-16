import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self, in_0, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9):
        tmp_0 = in_0
        tmp_1 = w_0
        tmp_2 = w_1
        tmp_3 = w_2
        tmp_4 = w_3
        tmp_5 = w_4
        tmp_6 = w_5
        tmp_7 = w_6
        tmp_8 = w_7
        tmp_9 = w_8
        tmp_10 = w_9
        tmp_11 = tmp_0.__eq__(1)
        tmp_12 = tmp_11.to(torch.float32)
        tmp_11 = None
        tmp_12 *= -3.4028234663852886e+38
        tmp_13 = tmp_12
        tmp_12 = None
        tmp_14 = tmp_13.unsqueeze(1)
        tmp_13 = None
        tmp_15 = tmp_14.unsqueeze(1)
        tmp_14 = None
        tmp_16 = torch.nn.functional.embedding(tmp_0, tmp_4, 1, None, 2.0, False, False)
        tmp_0 = tmp_4 = None
        tmp_17 = torch.ones((1, 15), dtype=torch.int64, device=device(type='cuda', index=0))
        tmp_18 = torch.cumsum(tmp_17, dim=1)
        tmp_19 = tmp_18 - tmp_17
        tmp_18 = tmp_17 = None
        tmp_19 += 2
        tmp_20 = tmp_19
        tmp_19 = None
        tmp_21 = torch.nn.functional.embedding(tmp_20, tmp_3, 1, None, 2.0, False, False)
        tmp_20 = tmp_3 = None
        tmp_22 = tmp_16 + tmp_21
        tmp_16 = tmp_21 = None
        tmp_23 = torch.nn.functional.layer_norm(tmp_22, (768,), tmp_2, tmp_1, 1e-05)
        tmp_22 = tmp_2 = tmp_1 = None
        tmp_24 = torch.nn.functional.dropout(tmp_23, 0.1, False, False)
        tmp_23 = None
        tmp_25 = torch.nn.functional.linear(tmp_24, tmp_8, tmp_7)
        tmp_8 = tmp_7 = None
        tmp_26 = torch.nn.functional.linear(tmp_24, tmp_6, tmp_5)
        tmp_6 = tmp_5 = None
        tmp_27 = tmp_26.view((1, 15, 12, 64))
        tmp_26 = None
        tmp_28 = tmp_27.permute(0, 2, 1, 3)
        tmp_27 = None
        tmp_29 = torch.nn.functional.linear(tmp_24, tmp_10, tmp_9)
        tmp_10 = tmp_9 = None
        tmp_30 = tmp_29.view((1, 15, 12, 64))
        tmp_29 = None
        tmp_31 = tmp_30.permute(0, 2, 1, 3)
        tmp_30 = None
        tmp_32 = tmp_25.view((1, 15, 12, 64))
        tmp_25 = None
        tmp_33 = tmp_32.permute(0, 2, 1, 3)
        tmp_32 = None
        tmp_34 = tmp_28.transpose(-1, -2)
        tmp_28 = None
        tmp_35 = torch.matmul(tmp_33, tmp_34)
        tmp_33 = tmp_34 = None
        tmp_36 = tmp_35 / 8.0
        tmp_35 = None
        tmp_37 = tmp_36 + tmp_15
        tmp_36 = None
        tmp_38 = torch.nn.functional.softmax(tmp_37, dim=-1)
        tmp_37 = None
        tmp_39 = torch.nn.functional.dropout(tmp_38, 0.1, False, False)
        tmp_38 = None
        tmp_40 = torch.matmul(tmp_39, tmp_31)
        tmp_39 = tmp_31 = None
        tmp_41 = tmp_40.permute(0, 2, 1, 3)
        tmp_40 = None
        tmp_42 = tmp_41.contiguous()
        tmp_41 = None
        tmp_43 = tmp_42.view((1, 15, 768))
        tmp_42 = None
        return (tmp_43, tmp_24, tmp_15)