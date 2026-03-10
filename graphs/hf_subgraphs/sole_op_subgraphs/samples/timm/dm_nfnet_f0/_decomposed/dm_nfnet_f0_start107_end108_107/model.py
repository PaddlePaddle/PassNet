import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.mul_(1.7015043497085571)
        return (tmp_0,)