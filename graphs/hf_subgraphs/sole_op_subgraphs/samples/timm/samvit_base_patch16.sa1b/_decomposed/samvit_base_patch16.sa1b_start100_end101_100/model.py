import torch

class GraphModule(torch.nn.Module):

    def forward(self):
        tmp_0 = torch.arange(14)
        return (tmp_0,)