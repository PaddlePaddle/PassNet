import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = torch.nn.functional.dropout(in_2, 0.0, False, False)
        tmp_3 = 0.0 + tmp_2
        tmp_2 = None
        tmp_4 = in_3 + tmp_3
        tmp_3 = None
        tmp_5 = torch.nn.functional.layer_norm(tmp_4, (512,), tmp_1, tmp_0, 1e-05)
        tmp_1 = tmp_0 = None
        return (tmp_5, tmp_4)