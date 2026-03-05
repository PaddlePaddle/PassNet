import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = in_0
        tmp_1 = in_1 * 0.448
        tmp_2 = tmp_1 + -0.08799999999999997
        tmp_1 = None
        tmp_3 = tmp_0[slice(None, None, None), 2]
        tmp_0 = None
        tmp_4 = torch.unsqueeze(tmp_3, 1)
        tmp_3 = None
        return (tmp_4, tmp_2)