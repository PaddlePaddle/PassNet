import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = in_0.sum(1)
        tmp_1 = torch.clamp(tmp_0, min=1e-09)
        tmp_0 = None
        tmp_2 = in_1 / tmp_1
        tmp_1 = None
        tmp_3 = torch.cat([tmp_2], 1)
        tmp_2 = None
        return (tmp_3,)