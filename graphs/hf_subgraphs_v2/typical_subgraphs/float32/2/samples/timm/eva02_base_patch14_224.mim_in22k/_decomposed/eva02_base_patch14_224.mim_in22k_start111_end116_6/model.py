import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = torch.nn.functional.linear(in_4, tmp_1, tmp_0)
        tmp_1 = tmp_0 = None
        tmp_5 = torch.nn.functional.silu(in_5, inplace=False)
        tmp_6 = tmp_5 * tmp_4
        tmp_5 = tmp_4 = None
        tmp_7 = torch.nn.functional.dropout(tmp_6, 0.0, False, False)
        tmp_6 = None
        tmp_8 = torch.nn.functional.layer_norm(tmp_7, (2048,), tmp_3, tmp_2, 1e-06)
        tmp_7 = tmp_3 = tmp_2 = None
        return (tmp_8,)