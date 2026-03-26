import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, in_0, in_1):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = w_4
        tmp_5 = w_5
        tmp_6 = w_6
        tmp_7 = w_7
        tmp_8 = in_0
        tmp_9 = in_1
        tmp_10 = torch.nn.functional.dropout(tmp_8, p=0.1, training=False)
        tmp_8 = None
        tmp_11 = tmp_9 + tmp_10
        tmp_9 = tmp_10 = None
        tmp_12 = torch.nn.functional.layer_norm(tmp_11, (256,), tmp_1, tmp_0, 1e-05)
        tmp_11 = tmp_1 = tmp_0 = None
        tmp_13 = torch.nn.functional.linear(tmp_12, tmp_3, tmp_2)
        tmp_3 = tmp_2 = None
        tmp_14 = torch.nn.functional.relu(tmp_13, inplace=False)
        tmp_13 = None
        tmp_15 = torch.nn.functional.dropout(tmp_14, p=0.0, training=False)
        tmp_14 = None
        tmp_16 = torch.nn.functional.linear(tmp_15, tmp_5, tmp_4)
        tmp_15 = tmp_5 = tmp_4 = None
        tmp_17 = torch.nn.functional.dropout(tmp_16, p=0.1, training=False)
        tmp_16 = None
        tmp_18 = tmp_12 + tmp_17
        tmp_12 = tmp_17 = None
        tmp_19 = torch.nn.functional.layer_norm(tmp_18, (256,), tmp_7, tmp_6, 1e-05)
        tmp_18 = tmp_7 = tmp_6 = None
        return (tmp_19,)