import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = torch.nn.functional.linear(in_5, tmp_3, tmp_2)
        tmp_3 = tmp_2 = None
        tmp_5 = torch.nn.functional.dropout(tmp_4, 0.1, False, False)
        tmp_4 = None
        tmp_6 = tmp_5 + in_4
        tmp_5 = None
        tmp_7 = torch.nn.functional.layer_norm(tmp_6, (256,), tmp_1, tmp_0, 1e-12)
        tmp_6 = tmp_1 = tmp_0 = None
        return (tmp_7,)