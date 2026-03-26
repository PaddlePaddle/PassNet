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
        tmp_8 = torch.nn.functional.linear(in_1, tmp_7, tmp_6)
        tmp_7 = tmp_6 = None
        tmp_9 = torch.nn.functional.dropout(tmp_8, 0.1, False, False)
        tmp_8 = None
        tmp_10 = tmp_9 + in_0
        tmp_9 = None
        tmp_11 = torch.nn.functional.layer_norm(tmp_10, (768,), tmp_5, tmp_4, 1e-12)
        tmp_10 = tmp_5 = tmp_4 = None
        tmp_12 = tmp_11[slice(None, None, None), 0, slice(None, None, None)]
        tmp_11 = None
        tmp_13 = torch.nn.functional.dropout(tmp_12, 0.1, False, False)
        tmp_12 = None
        tmp_14 = torch.nn.functional.linear(tmp_13, tmp_1, tmp_0)
        tmp_13 = tmp_1 = tmp_0 = None
        tmp_15 = torch.nn.functional.gelu(tmp_14)
        tmp_14 = None
        tmp_16 = torch.nn.functional.dropout(tmp_15, 0.1, False, False)
        tmp_15 = None
        tmp_17 = torch.nn.functional.linear(tmp_16, tmp_3, tmp_2)
        tmp_16 = tmp_3 = tmp_2 = None
        return (tmp_17,)