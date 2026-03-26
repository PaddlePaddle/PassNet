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
        tmp_8 = torch.nn.functional.gelu(in_1)
        tmp_9 = torch.nn.functional.linear(tmp_8, tmp_3, tmp_2)
        tmp_8 = tmp_3 = tmp_2 = None
        tmp_10 = torch.nn.functional.dropout(tmp_9, 0.1, False, False)
        tmp_9 = None
        tmp_11 = tmp_10 + in_0
        tmp_10 = None
        tmp_12 = torch.nn.functional.layer_norm(tmp_11, (312,), tmp_1, tmp_0, 1e-12)
        tmp_11 = tmp_1 = tmp_0 = None
        tmp_13 = tmp_12[slice(None, None, None), 0]
        tmp_12 = None
        tmp_14 = torch.nn.functional.linear(tmp_13, tmp_5, tmp_4)
        tmp_13 = tmp_5 = tmp_4 = None
        tmp_15 = torch.tanh(tmp_14)
        tmp_14 = None
        tmp_16 = torch.nn.functional.dropout(tmp_15, 0.1, False, False)
        tmp_15 = None
        tmp_17 = torch.nn.functional.linear(tmp_16, tmp_7, tmp_6)
        tmp_16 = tmp_7 = tmp_6 = None
        return (tmp_17,)