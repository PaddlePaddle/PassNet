import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, in_0, in_1):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = w_4
        tmp_5 = w_5
        tmp_6 = 1.702 * in_1
        tmp_7 = torch.sigmoid(tmp_6)
        tmp_6 = None
        tmp_8 = in_1 * tmp_7
        tmp_7 = None
        tmp_9 = torch.nn.functional.dropout(tmp_8, 0.0, False, False)
        tmp_8 = None
        tmp_10 = torch.nn.functional.linear(tmp_9, tmp_1, tmp_0)
        tmp_9 = tmp_1 = tmp_0 = None
        tmp_11 = torch.nn.functional.dropout(tmp_10, 0.0, False, False)
        tmp_10 = None
        tmp_12 = in_0 + tmp_11
        tmp_11 = None
        tmp_13 = torch.nn.functional.layer_norm(tmp_12, (1024,), tmp_5, tmp_4, 1e-05)
        tmp_12 = tmp_5 = tmp_4 = None
        tmp_14 = tmp_13[slice(None, None, None), 0]
        tmp_13 = None
        tmp_15 = torch.nn.functional.dropout(tmp_14, 0.0, False, False)
        tmp_14 = None
        tmp_16 = torch.nn.functional.linear(tmp_15, tmp_3, tmp_2)
        tmp_15 = tmp_3 = tmp_2 = None
        return (tmp_16,)