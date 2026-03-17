import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, in_0, in_1, in_2):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = in_2.view((1, 27, 12, -1))
        tmp_5 = tmp_4.permute(0, 2, 1, 3)
        tmp_4 = None
        tmp_6 = tmp_5.contiguous()
        tmp_5 = None
        tmp_7 = tmp_6.view(-1, 27, 64)
        tmp_6 = None
        tmp_8 = torch.nn.functional.linear(in_1, tmp_1, tmp_0)
        tmp_1 = tmp_0 = None
        tmp_9 = tmp_8.view((1, 27, 12, -1))
        tmp_8 = None
        tmp_10 = tmp_9.permute(0, 2, 1, 3)
        tmp_9 = None
        tmp_11 = tmp_10.contiguous()
        tmp_10 = None
        tmp_12 = tmp_11.view(-1, 27, 64)
        tmp_11 = None
        tmp_13 = torch.nn.functional.linear(in_1, tmp_3, tmp_2)
        tmp_3 = tmp_2 = None
        tmp_14 = tmp_13.view((1, 27, 12, -1))
        tmp_13 = None
        tmp_15 = tmp_14.permute(0, 2, 1, 3)
        tmp_14 = None
        tmp_16 = tmp_15.contiguous()
        tmp_15 = None
        tmp_17 = tmp_16.view(-1, 27, 64)
        tmp_16 = None
        tmp_18 = torch.tensor(64, dtype=torch.float32)
        tmp_19 = tmp_18 * 1
        tmp_18 = None
        tmp_20 = torch.sqrt(tmp_19)
        tmp_19 = None
        tmp_21 = tmp_12.transpose(-1, -2)
        tmp_12 = None
        tmp_22 = tmp_20.to(dtype=torch.float32)
        tmp_20 = None
        tmp_23 = tmp_21 / tmp_22
        tmp_21 = tmp_22 = None
        tmp_24 = torch.bmm(tmp_7, tmp_23)
        tmp_7 = tmp_23 = None
        tmp_25 = tmp_24.view(-1, 12, 27, 27)
        tmp_24 = None
        tmp_26 = in_0.bool()
        tmp_27 = ~tmp_26
        tmp_26 = None
        tmp_28 = tmp_25.masked_fill(tmp_27, -3.4028234663852886e+38)
        tmp_25 = tmp_27 = None
        tmp_29 = torch.nn.functional.softmax(tmp_28, dim=-1)
        tmp_28 = None
        tmp_30 = torch.nn.functional.dropout(tmp_29, 0.1, False, False)
        tmp_29 = None
        tmp_31 = tmp_30.view(-1, 27, 27)
        tmp_30 = None
        tmp_32 = torch.bmm(tmp_31, tmp_17)
        tmp_31 = tmp_17 = None
        tmp_33 = tmp_32.view(-1, 12, 27, 64)
        tmp_32 = None
        tmp_34 = tmp_33.permute(0, 2, 1, 3)
        tmp_33 = None
        tmp_35 = tmp_34.contiguous()
        tmp_34 = None
        tmp_36 = tmp_35.view((1, 27, -1))
        tmp_35 = None
        return (tmp_36,)