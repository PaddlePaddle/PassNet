import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = in_4
        tmp_5 = torch.nn.functional.linear(in_6, tmp_1, tmp_0)
        tmp_1 = tmp_0 = None
        tmp_6 = torch.nn.functional.dropout(tmp_5, 0.0, False, False)
        tmp_5 = None
        tmp_7 = in_5 + tmp_6
        tmp_6 = None
        tmp_8 = torch.nn.functional.layer_norm(tmp_7, (192,), tmp_4, tmp_3, 1e-06)
        tmp_4 = tmp_3 = None
        tmp_9 = torch.nn.functional.linear(tmp_8, tmp_2, None)
        tmp_8 = tmp_2 = None
        return (tmp_9, tmp_7)