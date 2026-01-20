import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.nn.functional.normalize(in_0, dim=-1)
        tmp_1 = tmp_0.transpose(-2, -1)
        tmp_0 = None
        return (tmp_1,)