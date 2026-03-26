import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.clamp(in_0, max=4.605170185988092)
        return (tmp_0,)