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
        tmp_7 = torch.nn.functional.linear(tmp_6, tmp_3, tmp_2)
        tmp_6 = tmp_3 = tmp_2 = None
        tmp_8 = torch.nn.functional.dropout(tmp_7, 0.0, False, False)
        tmp_7 = None
        tmp_9 = tmp_8 + in_0
        tmp_8 = None
        tmp_10 = torch.nn.functional.layer_norm(tmp_9, (768,), tmp_5, tmp_4, 1e-12)
        tmp_9 = tmp_5 = tmp_4 = None
        tmp_11 = tmp_10[slice(None, None, None), 0, slice(None, None, None)]
        tmp_10 = None
        tmp_12 = torch.nn.functional.linear(tmp_11, tmp_1, tmp_0)
        tmp_11 = tmp_1 = tmp_0 = None
        return (tmp_12,)