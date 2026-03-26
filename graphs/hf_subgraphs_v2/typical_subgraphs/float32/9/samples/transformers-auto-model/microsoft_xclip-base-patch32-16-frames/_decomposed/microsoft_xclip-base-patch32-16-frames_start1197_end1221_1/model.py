import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, in_0, in_1, in_2):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = w_4
        tmp_5 = w_5
        tmp_6 = w_6
        tmp_7 = 1.702 * in_0
        tmp_8 = torch.sigmoid(tmp_7)
        tmp_7 = None
        tmp_9 = in_0 * tmp_8
        tmp_8 = None
        tmp_10 = torch.nn.functional.dropout(tmp_9, 0.0, False, False)
        tmp_9 = None
        tmp_11 = torch.nn.functional.linear(tmp_10, tmp_1, tmp_0)
        tmp_10 = tmp_1 = tmp_0 = None
        tmp_12 = in_2 + tmp_11
        tmp_11 = None
        tmp_13 = torch.nn.functional.layer_norm(tmp_12, (512,), tmp_6, tmp_5, 1e-05)
        tmp_6 = tmp_5 = None
        tmp_14 = torch.nn.functional.linear(tmp_13, tmp_3, None)
        tmp_13 = tmp_3 = None
        tmp_15 = tmp_14.reshape(1, 1, 8, 64)
        tmp_14 = None
        tmp_16 = tmp_15.permute(0, 2, 1, 3)
        tmp_15 = None
        tmp_17 = torch.nn.functional.linear(in_1, tmp_2, None)
        tmp_2 = None
        tmp_18 = tmp_17.reshape(1, 49, 8, 64)
        tmp_17 = None
        tmp_19 = tmp_18.permute(0, 2, 1, 3)
        tmp_18 = None
        tmp_20 = torch.nn.functional.linear(in_1, tmp_4, None)
        tmp_4 = None
        tmp_21 = tmp_20.reshape(1, 49, 8, 64)
        tmp_20 = None
        tmp_22 = tmp_21.permute(0, 2, 1, 3)
        tmp_21 = None
        tmp_23 = tmp_19.transpose(-2, -1)
        tmp_19 = None
        tmp_24 = tmp_16 @ tmp_23
        tmp_16 = tmp_23 = None
        tmp_25 = tmp_24 * 0.125
        tmp_24 = None
        tmp_26 = tmp_25.softmax(dim=-1)
        tmp_25 = None
        tmp_27 = torch.nn.functional.dropout(tmp_26, 0.0, False, False)
        tmp_26 = None
        tmp_28 = tmp_27 @ tmp_22
        tmp_27 = tmp_22 = None
        tmp_29 = tmp_28.transpose(1, 2)
        tmp_28 = None
        tmp_30 = tmp_29.reshape(1, 1, 512)
        tmp_29 = None
        return (tmp_12, tmp_30)