import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, in_0, in_1):
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
        tmp_12 = torch.nn.functional.linear(in_0, tmp_10, tmp_9)
        tmp_10 = tmp_9 = None
        tmp_13 = torch.nn.functional.dropout(tmp_12, p=0.1, training=False)
        tmp_12 = None
        tmp_14 = in_1 + tmp_13
        tmp_13 = None
        tmp_15 = torch.nn.functional.layer_norm(tmp_14, (1024,), tmp_8, tmp_7, 1e-05)
        tmp_14 = tmp_8 = tmp_7 = None
        tmp_16 = tmp_15.reshape(-1, 1024)
        tmp_15 = None
        tmp_17 = torch.nn.functional.linear(tmp_16, tmp_2, tmp_1)
        tmp_2 = tmp_1 = None
        tmp_18 = torch.nn.functional.relu(tmp_17, inplace=False)
        tmp_17 = None
        tmp_19 = torch.nn.functional.linear(tmp_18, tmp_4, tmp_3)
        tmp_18 = tmp_4 = tmp_3 = None
        tmp_20 = torch.nn.functional.dropout(tmp_19, p=0.1, training=False)
        tmp_19 = None
        tmp_21 = tmp_16 + tmp_20
        tmp_16 = tmp_20 = None
        tmp_22 = tmp_21.view((1, 20, 1024))
        tmp_21 = None
        tmp_23 = torch.nn.functional.layer_norm(tmp_22, (1024,), tmp_6, tmp_5, 1e-05)
        tmp_22 = tmp_6 = tmp_5 = None
        tmp_24 = torch.nn.functional.linear(tmp_23, tmp_11, None)
        tmp_23 = tmp_11 = None
        tmp_25 = torch.nn.functional.linear(tmp_24, tmp_0, None)
        tmp_24 = tmp_0 = None
        tmp_26 = tmp_25.contiguous()
        tmp_25 = None
        return (tmp_26,)