import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, in_0, in_1):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = w_4
        tmp_5 = w_5
        tmp_6 = torch.nn.functional.gelu(in_1)
        tmp_7 = torch.nn.functional.dropout(tmp_6, 0.0, False, False)
        tmp_6 = None
        tmp_8 = torch.nn.functional.linear(tmp_7, tmp_1, tmp_0)
        tmp_7 = tmp_1 = tmp_0 = None
        tmp_9 = torch.nn.functional.dropout(tmp_8, 0.1, False, False)
        tmp_8 = None
        tmp_10 = in_0 + tmp_9
        tmp_9 = None
        tmp_11 = torch.nn.functional.layer_norm(tmp_10, (32,), tmp_3, tmp_2, 1e-05)
        tmp_10 = tmp_3 = tmp_2 = None
        tmp_12 = tmp_11[slice(None, None, None), 0]
        tmp_13 = torch.nn.functional.linear(tmp_12, tmp_5, tmp_4)
        tmp_12 = tmp_5 = tmp_4 = None
        tmp_14 = torch.tanh(tmp_13)
        tmp_13 = None
        return (tmp_11, tmp_14)