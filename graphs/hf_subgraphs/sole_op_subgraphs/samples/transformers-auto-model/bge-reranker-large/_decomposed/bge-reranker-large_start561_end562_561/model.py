import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.clamp(in_0, min=1e-09)
        return (tmp_0,)