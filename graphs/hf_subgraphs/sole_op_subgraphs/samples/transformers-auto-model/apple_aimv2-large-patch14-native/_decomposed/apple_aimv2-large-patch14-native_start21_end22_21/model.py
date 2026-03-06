import torch

class GraphModule(torch.nn.Module):

    def forward(self):
        tmp_0 = torch.arange(256)
        return (tmp_0,)