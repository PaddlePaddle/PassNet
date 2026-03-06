import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.stack([in_0], dim=-2)
        return (tmp_0,)