import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.div(in_0, 14, rounding_mode='trunc')
        return (tmp_0,)