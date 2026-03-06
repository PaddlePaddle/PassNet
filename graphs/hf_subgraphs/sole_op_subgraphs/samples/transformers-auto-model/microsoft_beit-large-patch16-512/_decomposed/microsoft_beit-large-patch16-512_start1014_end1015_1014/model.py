import torch

class GraphModule(torch.nn.Module):

    def forward(self):
        tmp_0 = torch.arange(32)
        return (tmp_0,)