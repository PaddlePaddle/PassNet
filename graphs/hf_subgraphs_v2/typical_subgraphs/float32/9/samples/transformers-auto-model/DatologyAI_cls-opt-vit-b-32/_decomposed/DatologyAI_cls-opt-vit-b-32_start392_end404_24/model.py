import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, in_0, in_1):
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
        tmp_10 = torch.nn.functional.linear(in_0, tmp_9, tmp_8)
        tmp_9 = tmp_8 = None
        tmp_11 = torch.nn.functional.dropout(tmp_10, p=0.1, training=False)
        tmp_10 = None
        tmp_12 = in_1 + tmp_11
        tmp_11 = None
        tmp_13 = tmp_12.reshape(-1, 768)
        tmp_12 = None
        tmp_14 = torch.nn.functional.layer_norm(tmp_13, (768,), tmp_7, tmp_6, 1e-05)
        tmp_7 = tmp_6 = None
        tmp_15 = torch.nn.functional.linear(tmp_14, tmp_3, tmp_2)
        tmp_14 = tmp_3 = tmp_2 = None
        tmp_16 = torch.nn.functional.relu(tmp_15, inplace=False)
        tmp_15 = None
        tmp_17 = torch.nn.functional.linear(tmp_16, tmp_5, tmp_4)
        tmp_16 = tmp_5 = tmp_4 = None
        tmp_18 = torch.nn.functional.dropout(tmp_17, p=0.1, training=False)
        tmp_17 = None
        tmp_19 = tmp_13 + tmp_18
        tmp_13 = tmp_18 = None
        tmp_20 = tmp_19.view((1, 13, 768))
        tmp_19 = None
        tmp_21 = torch.nn.functional.layer_norm(tmp_20, (768,), tmp_1, tmp_0, 1e-05)
        tmp_20 = tmp_1 = tmp_0 = None
        return (tmp_21,)