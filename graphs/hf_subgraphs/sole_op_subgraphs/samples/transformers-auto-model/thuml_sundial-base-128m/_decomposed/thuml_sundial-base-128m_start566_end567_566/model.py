import torch

class GraphModule(torch.nn.Module):

    def forward(self):
        tmp_0 = torch.randn(1, 720)
        return (tmp_0,)