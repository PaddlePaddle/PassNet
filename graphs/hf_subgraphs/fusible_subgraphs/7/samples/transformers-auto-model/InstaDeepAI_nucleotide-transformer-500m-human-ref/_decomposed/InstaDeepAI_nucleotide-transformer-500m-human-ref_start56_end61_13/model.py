import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0 * 0.5
        tmp_1 = in_0 / 1.4142135623730951
        tmp_2 = torch.erf(tmp_1)
        tmp_1 = None
        tmp_3 = 1.0 + tmp_2
        tmp_2 = None
        tmp_4 = tmp_0 * tmp_3
        tmp_0 = tmp_3 = None
        return (tmp_4,)